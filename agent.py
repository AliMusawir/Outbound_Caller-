from __future__ import annotations

import asyncio
import logging
from dotenv import load_dotenv
import json
import os
from datetime import datetime
from typing import Any
import subprocess
import time

from livekit import rtc, api
from livekit.agents import (
    AgentSession,
    Agent,
    JobContext,
    function_tool,
    RunContext,
    get_job_context,
    cli,
    WorkerOptions,
    RoomInputOptions,
)
from livekit.plugins import (
    deepgram,
    google,
    silero,
    noise_cancellation,
)

# Load environment variables
import pathlib
current_dir = pathlib.Path(__file__).parent
env_path = current_dir / ".env.local"

logger = logging.getLogger("outbound-caller")
logger.setLevel(logging.INFO)

load_dotenv(dotenv_path=str(env_path))

# Load all required environment variables
outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")
deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
outbound_call_timeout_s = int(os.getenv("OUTBOUND_CALL_TIMEOUT", "45"))
outbound_retry_count = int(os.getenv("OUTBOUND_RETRY_COUNT", "1"))
outbound_retry_delay_s = int(os.getenv("OUTBOUND_RETRY_DELAY", "30"))
gcs_bucket = os.getenv("GCS_BUCKET")
gcp_credentials_json = os.getenv("GCP_CREDENTIALS_JSON")

# Validate required environment variables
if not deepgram_api_key:
    logger.error("DEEPGRAM_API_KEY is required")
    raise ValueError("DEEPGRAM_API_KEY environment variable is required")

if not google_api_key:
    logger.error("GOOGLE_API_KEY is required")
    raise ValueError("GOOGLE_API_KEY environment variable is required")

# Validate SIP trunk id
if not outbound_trunk_id:
    logger.error("SIP_OUTBOUND_TRUNK_ID is required")
    raise ValueError("SIP_OUTBOUND_TRUNK_ID environment variable is required")

def save_no_answer_note(phone_number: str, reason: str, dial_info: dict[str, Any] | None = None) -> str | None:
    """Save a JSON call note for no-answer/failed call attempts."""
    try:
        import pathlib
        notes_dir = pathlib.Path(__file__).parent / "call_notes"
        notes_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().isoformat()
        filename = f"call_notes_{phone_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = notes_dir / filename

        safe_dial = dial_info or {}
        note = {
            "phone_number": phone_number,
            "timestamp": timestamp,
            "call_duration": "not_connected",
            "status": "no_answer",
            "patient_info": {
                "name": safe_dial.get("patient_name", ""),
                "appointment_date": "",
                "reason_for_visit": safe_dial.get("notes", ""),
                "medications": "",
                "emergency_contact": "",
                "additional_symptoms": "",
                "additional_info": ""
            },
            "agent_notes": f"Call not connected. Reason: {reason}",
            "priority": safe_dial.get("priority", "normal")
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(note, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved no-answer note to: {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Failed to save no-answer note: {e}")
        return None

class OutboundCaller(Agent):
    def __init__(
        self,
        *,
        name: str,
        appointment_time: str,
        dial_info: dict[str, Any],
    ):
        # Build system instructions. If a doctor note is present, override to be note-driven.
        doctor_note_for_prompt = (dial_info.get("doctor_note") or "").strip()
        # Keep a copy for reporting/summary logic
        self.doctor_note: str = doctor_note_for_prompt
        self.note_mode: bool = bool(self.doctor_note)
        self.ready_to_end: bool = False
        # If note provided, infer a chief complaint from the first clause if not set later
        def _infer_chief_from_note(note: str) -> str:
            text = (note or "").strip()
            if not text:
                return ""
            import re
            # take up to first semicolon or sentence end
            m = re.split(r"[;\n]+|(?<=[.?!])\s+", text, maxsplit=1)
            first = (m[0] if m else text).strip()
            # remove trailing report labels
            first = first.replace("PRE-VISIT MEDICAL INTAKE REPORT", "").strip()
            return first[:120]
        self.inferred_chief_from_note: str = _infer_chief_from_note(self.doctor_note) if self.note_mode else ""
        if doctor_note_for_prompt:
            instructions = f"""
            You are a professional, empathetic medical intake specialist on a phone call.
            For THIS CALL, interview ONLY from the provider note below. Do not add topics.
            Ask one brief question at a time and leave space for answers.

            Provider note (sole source of topics):
            {doctor_note_for_prompt}

            Style:
            - Warm, natural phrasing (≤ 15 words per sentence)
            - Use brief, varied backchannels sparingly (e.g., "okay", "hmm", "I see"); avoid repeating the same phrase
            - Avoid lists or multi‑part questions; no reading the note aloud
            - Stay strictly in intake scope; do not explain LLM capabilities

            Flow:
            1) Say you’ll ask a few questions based on the provider’s note
            2) Ask the prepared questions sequentially in plain language
            3) After each answer, record it with record_patient_info (best category/subcategory)
            4) When finished, call summarize_and_confirm and wait for explicit yes/no before end_call

            Rules:
            - No unrelated topics
            - One concise question at a time
            - Keep it conversational and professional
            - Do not end the call unless the user explicitly confirms or asks to end
            - For the final summary, produce ONE concise paragraph that reflects the conversation
              (no markdown headings, no 4-section template). The 4-section template is used ONLY when there is NO provider note.
            - After EVERY user reply, extract concrete facts and IMMEDIATELY call
              record_patient_info with the proper info_type/subcategory before asking
              the next question. Avoid replying with only acknowledgments.
            - Avoid using or mentioning templates or headings in your final summary; it must be a natural paragraph.
            """
        else:
            instructions = f"""
            You are a professional medical intake specialist conducting pre-visit screening calls. 
            Follow medical interview protocols and generate comprehensive clinical documentation.
            
            MEDICAL INTERVIEW PROTOCOL - STRUCTURED REPORTING (Maximum 20 questions):
            
            Your responses must follow this EXACT format for the final summary:
            
            ### Primary concern:
            [Collect and list the primary concern/chief complaint the patient is having]
            
            ### History of Present Illness (HPI):
            [Probe deeply to collect comprehensive HPI. Ask about: when it started, progression, triggers, relieving factors, severity, timing, duration, associated symptoms]
            
            ### Relevant Medical History (from EHR):
            [Extract relevant past medical conditions, surgeries, or family history that relates to current complaint]
            
            ### Medications (from EHR and interview):
            [Document current medications, dosages, allergies, and any new medications mentioned]
            
            INTERVIEW STRATEGY:
            - Ask focused questions to fill each section completely
            - Use record_patient_info tool for each piece of information
            - Prioritize information that fits these 4 sections
            - Maximum 20 questions to complete all sections
            - Be thorough but respectful of patient's time
            
            CRITICAL RULES:
            - Ask only ONE question at a time
            - Use medical terminology when appropriate
            - Record ALL information using record_patient_info tool with proper subcategories
            - Focus on gathering information for the 4 structured sections
            - End with summarize_and_confirm when complete
            
            DATA RECORDING INSTRUCTIONS:
            - Use record_patient_info with info_type="chief_complaint" for main concern
            - Use record_patient_info with info_type="hpi" and subcategory="onset" for when it started
            - Use record_patient_info with info_type="hpi" and subcategory="quality" for pain description
            - Use record_patient_info with info_type="hpi" and subcategory="severity" for pain scale
            - Use record_patient_info with info_type="hpi" and subcategory="timing" for when it occurs
            - Use record_patient_info with info_type="hpi" and subcategory="radiation" for pain spread
            - Use record_patient_info with info_type="medications" for current medications
            - Use record_patient_info with info_type="allergies" for allergies
            - Use record_patient_info with info_type="pmh" for past medical history
            - Use record_patient_info with info_type="family" for family history
            
            TOOLS TO USE:
            - record_patient_info: Record each piece of information immediately with proper subcategories
            - summarize_and_confirm: Generate final summary and confirmation
            - end_call: End call after confirmation
            - transfer_call: If patient requests human agent
            
            The patient's name is {name}. Their appointment is on {appointment_time}.
            """
        super().__init__(instructions=instructions)
        # keep reference to the participant for transfers
        self.participant: rtc.RemoteParticipant | None = None
        self.dial_info = dial_info
        
        # Enhanced patient information structure for medical reports
        self.patient_info = {
            "name": "",
            "appointment_date": "",
            "chief_complaint": "",
            "history_of_present_illness": {
                "onset": "",
                "provocation": "",
                "quality": "",
                "radiation": "",
                "severity": "",
                "timing": "",
                "duration": ""
            },
            "review_of_systems": {
                "constitutional": "",
                "cardiovascular": "",
                "respiratory": "",
                "gastrointestinal": "",
                "musculoskeletal": "",
                "neurological": "",
                "psychiatric": "",
                "pertinent_negatives": []
            },
            "medications": "",
            "allergies": "",
            "past_medical_history": "",
            "social_history": "",
            "family_history": "",
            "additional_notes": ""
        }
        
        # Template support removed
        self.template_opening = ""
        self.template_required = []
        self.template_followups = []
        self.template_closing = ""
        
        # Track interview progress
        self.interview_phase = "identification"
        self.question_count = 0
        self.max_questions = 20
        
        # Store call summary
        self.call_summary = None
        # Connection state
        self.connected = False
        # Egress/recording state
        self.recording_filename = None
        self.recording_gcs_uri = None
        self.recording_local_path = None

        # Template mode removed
        self.template_mode = False
        self.template_required_index = 0
        self.template_followup_index = 0
        self.last_question = None
        self.qa_log: list[dict[str, str]] = []
        # Planning guard to prevent tool execution during internal planning
        self.planning_mode: bool = False
        # Confirmation flow state
        self.awaiting_confirmation: bool = False
        self.user_confirmed: bool = False
        self.summary_done: bool = False
        # Identification steps
        self.id_name_done: bool = False
        self.id_chief_done: bool = False

    def _count_recorded_items(self) -> int:
        """Count how many structured fields have been recorded to estimate progress."""
        count = 0
        # Top-level fields
        for key in ["chief_complaint", "medications", "allergies", "past_medical_history", "name", "appointment_date"]:
            val = self.patient_info.get(key)
            if isinstance(val, str) and val.strip():
                count += 1
        # HPI subfields
        for key, val in self.patient_info.get("history_of_present_illness", {}).items():
            if isinstance(val, str) and val.strip():
                count += 1
        # ROS subfields (ignore list of negatives for counting simplicity)
        for key, val in self.patient_info.get("review_of_systems", {}).items():
            if key == "pertinent_negatives":
                continue
            if isinstance(val, str) and val.strip():
                count += 1
        return count

    def set_participant(self, participant: rtc.RemoteParticipant):
        self.participant = participant
        self.connected = True

    async def hangup(self):
        """Helper function to hang up the call by deleting the room"""

        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(
            api.DeleteRoomRequest(
                room=job_ctx.room.name,
            )
        )

    @function_tool()
    async def transfer_call(self, ctx: RunContext):
        """Transfer the call to a human agent, called after confirming with the user"""

        transfer_to = self.dial_info["transfer_to"]
        if not transfer_to:
            return "cannot transfer call"

        logger.info(f"transferring call to {transfer_to}")

        # let the message play fully before transferring
        await ctx.session.generate_reply(
            instructions="let the user know you'll be transferring them"
        )

        job_ctx = get_job_context()
        try:
            await ctx.api.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    room_name=job_ctx.room.name,
                    participant_identity=self.participant.identity,
                    transfer_to=f"tel:{transfer_to}",
                )
            )

            logger.info(f"transferred call to {transfer_to}")
        except Exception as e:
            logger.error(f"error transferring call: {e}")
            await ctx.session.generate_reply(
                instructions="there was an error transferring the call."
            )
            await self.hangup()

    @function_tool()
    async def end_call(self, ctx: RunContext):
        """Called when the user wants to end the call"""
        who = self.participant.identity if self.participant else 'unknown'
        logger.info(f"ending the call for {who}")

        # Prevent premature end in note-mode unless a summary has been delivered
        if getattr(self, 'note_mode', False) and not getattr(self, 'ready_to_end', False):
            logger.info("Blocking premature end_call: waiting for summary/confirmation in note-mode")
            await ctx.session.generate_reply(
                instructions="Politely explain you'll continue with a couple of brief questions and summarize before ending."
            )
            return

        # Generate final call summary
        final_summary = {
            "patient_info": self.patient_info,
            "call_duration": "completed",
            "status": "call_ended",
            "timestamp": datetime.now().isoformat(),
            "question_count": self.question_count
        }
        
        # If no questions were asked and this was note-driven, include a concise explanation
        if getattr(self, 'doctor_note', None) and self._count_recorded_items() == 0:
            final_summary["status"] = "call_ended_no_answers_note_mode"

        logger.info(f"Final call summary: {final_summary}")
        
        # Save professional medical report to file
        await self.save_medical_report(final_summary)
        
        # let the agent finish speaking
        await ctx.wait_for_playout()

        await self.hangup()

    def get_call_summary(self):
        """Get the current call summary"""
        if self.call_summary:
            return self.call_summary
        else:
            # Generate a basic summary from patient info
            return {
                "patient_info": self.patient_info.copy(),
                "call_duration": "in_progress",
                "status": "no_summary_generated",
                "timestamp": datetime.now().isoformat(),
                "question_count": self.question_count
            }

    async def save_medical_report(self, call_summary):
        """Save professional medical report to a TXT file"""
        try:
            # Create call_notes directory if it doesn't exist
            import pathlib
            notes_dir = pathlib.Path(__file__).parent / "call_notes"
            notes_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp and phone number
            phone_number = self.participant.identity if self.participant else "unknown"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"medical_report_{phone_number}_{timestamp}.txt"
            filepath = notes_dir / filename
            
            # Generate professional medical report
            medical_report = self.generate_medical_report()
            
            # Create comprehensive text report
            # Prefer a computed question metric if the tracked count is zero
            computed_items = self._count_recorded_items()
            effective_questions = max(call_summary["question_count"], computed_items)

            text_report = f"""
{medical_report}

CALL INFORMATION:
Phone Number: {phone_number}
Call Date: {call_summary["timestamp"]}
Call Duration: {call_summary["call_duration"]}
Call Status: {call_summary["status"]}
Questions Asked: {effective_questions}
Report Type: Pre-visit Medical Screening
Priority: Normal

PATIENT SUMMARY:
Name: {call_summary["patient_info"].get("name", "Not provided")}
Appointment Date: {call_summary["patient_info"].get("appointment_date", "Not provided")}
Chief Complaint: {call_summary["patient_info"].get("chief_complaint", "Not specified")}

NOTES:
This medical intake report was generated during a pre-visit screening call. The information collected helps healthcare providers prepare for the patient's appointment by understanding their current symptoms, medical history, and medication needs. All information should be verified during the actual medical visit.

Report generated by AI Medical Intake Specialist
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            # Save to TXT file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text_report.strip())
            
            logger.info(f"Medical report saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving medical report: {e}")
            return None

    def generate_medical_report(self):
        """Generate a professional medical report in paragraph format.
        If a doctor note is present and little/no patient info was collected,
        generate a concise note-driven paragraph instead of the full template.
        """
        # If a doctor note is present, always produce a single-paragraph conversation summary
        minimal_info = not (
            self.patient_info.get('chief_complaint')
            or any(self.patient_info['history_of_present_illness'].values())
            or self.patient_info.get('medications')
            or self.patient_info.get('allergies')
            or self.patient_info.get('past_medical_history')
        )
        if self.doctor_note:
            summary_parts: list[str] = []
            if self.patient_info['name']:
                summary_parts.append(f"Patient identified as {self.patient_info['name']}.")
            chief = self.patient_info.get('chief_complaint') or getattr(self, 'inferred_chief_from_note', '')
            if chief:
                summary_parts.append(f"Primary concern is {chief}.")
            hpi = self.patient_info['history_of_present_illness']
            hpi_bits: list[str] = []
            if hpi.get('quality'):
                hpi_bits.append(hpi['quality'])
            if hpi.get('onset'):
                hpi_bits.append(f"onset {hpi['onset']}")
            if hpi.get('severity'):
                hpi_bits.append(f"severity {hpi['severity']}")
            if hpi.get('duration'):
                hpi_bits.append(f"duration {hpi['duration']}")
            if hpi.get('timing'):
                hpi_bits.append(f"timing {hpi['timing']}")
            if hpi.get('radiation'):
                hpi_bits.append(f"radiation {hpi['radiation']}")
            if hpi_bits:
                summary_parts.append("History of present illness: " + ", ".join(hpi_bits) + ".")
            if self.patient_info.get('past_medical_history'):
                summary_parts.append(f"Relevant history includes {self.patient_info['past_medical_history']}.")
            if self.patient_info.get('medications'):
                summary_parts.append(f"Current medications include {self.patient_info['medications']}.")
            if self.patient_info.get('allergies'):
                summary_parts.append(f"Allergies: {self.patient_info['allergies']}.")
            ros = self.patient_info.get('review_of_systems', {})
            ros_bits: list[str] = []
            for key in ['respiratory', 'cardiovascular', 'gastrointestinal', 'musculoskeletal', 'neurological', 'psychiatric']:
                val = ros.get(key)
                if isinstance(val, str) and val.strip():
                    # Clean internal tags like "obstructive_sleep_apnea_risk_factors: "
                    clean_val = val.replace("obstructive_sleep_apnea_risk_factors: ", "").strip()
                    ros_bits.append(f"{key.capitalize()}: {clean_val}")
            if ros_bits:
                summary_parts.append("Review of systems: " + "; ".join(ros_bits) + ".")
            if self.patient_info.get('additional_notes'):
                summary_parts.append(self.patient_info['additional_notes'].replace("past_medical_history (anesthesia_problems): ", "Anesthesia: "))
            if minimal_info and not summary_parts:
                summary_parts.append("No patient responses were captured during this call.")
            # Build paragraph using conversation-derived fields only
            paragraph = " ".join(summary_parts).strip()
            import textwrap
            wrapped_paragraph = textwrap.fill(paragraph, width=92)

            effective_q = max(self.question_count, self._count_recorded_items())
            report = f"""
PRE-VISIT MEDICAL INTAKE REPORT

Note-Driven Conversation Summary:
{wrapped_paragraph}

Report Details:
Questions Asked: {effective_q}
Interview Status: {self.interview_phase}
            """
            return report.strip()
        # Incorporate QA log if present (template mode)
        qa_section = ""
        if getattr(self, 'qa_log', None):
            pairs = [f"- {p.get('question')}: {p.get('answer')}" for p in self.qa_log]
            qa_section = "\nTemplate Q&A Summary:\n" + "\n".join(pairs)
        
        # Build HPI paragraph
        hpi_parts = []
        if self.patient_info['history_of_present_illness']['onset']:
            hpi_parts.append(f"The patient reports that symptoms {self.patient_info['history_of_present_illness']['onset']}")
        if self.patient_info['history_of_present_illness']['quality']:
            hpi_parts.append(f"The patient describes the pain as {self.patient_info['history_of_present_illness']['quality']}")
        if self.patient_info['history_of_present_illness']['severity']:
            hpi_parts.append(f"Pain severity is rated {self.patient_info['history_of_present_illness']['severity']}")
        if self.patient_info['history_of_present_illness']['timing']:
            hpi_parts.append(f"The pain {self.patient_info['history_of_present_illness']['timing']}")
        if self.patient_info['history_of_present_illness']['radiation']:
            hpi_parts.append(f"Pain radiates {self.patient_info['history_of_present_illness']['radiation']}")
        
        hpi_text = '. '.join(hpi_parts) if hpi_parts else 'Limited information available about the history of present illness.'
        
        # Build medical history paragraph
        medical_history_parts = []
        if self.patient_info['past_medical_history']:
            medical_history_parts.append(f"Past medical history includes {self.patient_info['past_medical_history']}")
        if self.patient_info['family_history']:
            medical_history_parts.append(f"Family history reveals {self.patient_info['family_history']}")
        
        medical_history_text = '. '.join(medical_history_parts) if medical_history_parts else 'No significant past medical history reported.'
        
        # Build medications paragraph
        medications_parts = []
        if self.patient_info['medications']:
            medications_parts.append(f"Current medications include {self.patient_info['medications']}")
        if self.patient_info['allergies']:
            medications_parts.append(f"The patient reports {self.patient_info['allergies']}")
        
        medications_text = '. '.join(medications_parts) if medications_parts else 'No medications reported.'
        
        # Build a compact report including only fields that are present
        report_lines = [
            "PRE-VISIT MEDICAL INTAKE REPORT",
            "",
        ]
        if self.patient_info['name'] or self.patient_info['appointment_date']:
            report_lines.extend([
                "Patient Information:",
                f"Patient Name: {self.patient_info['name'] or 'Not provided'}",
                f"Appointment Date: {self.patient_info['appointment_date'] or 'Not provided'}",
                f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "",
            ])
        chief = self.patient_info['chief_complaint'] or self.inferred_chief_from_note
        if chief:
            report_lines.extend([
                "Primary Concern:",
                chief,
                "",
            ])
        if hpi_text and hpi_text.strip() and hpi_text != 'Limited information available about the history of present illness.':
            report_lines.extend([
                "History of Present Illness (HPI):",
                hpi_text,
                "",
            ])
        if medical_history_text and medical_history_text.strip() and medical_history_text != 'No significant past medical history reported.':
            report_lines.extend([
                "Relevant Medical History:",
                medical_history_text,
                "",
            ])
        if medications_text and medications_text.strip() and medications_text != 'No medications reported.':
            report_lines.extend([
                "Current Medications and Allergies:",
                medications_text,
                "",
            ])
        if qa_section:
            report_lines.extend([qa_section, ""])
        effective_q = max(self.question_count, self._count_recorded_items())
        report_lines.extend([
            "Report Details:",
            f"Questions Asked: {effective_q}",
            f"Interview Status: {self.interview_phase}",
        ])
        report = "\n".join(report_lines)
        return report.strip()

    @function_tool()
    async def record_patient_info(self, ctx: RunContext, info_type: str, value: str, subcategory: str = None):
        """Record specific patient information during the medical interview
        
        Args:
            info_type: Type of information (name, appointment_date, chief_complaint, hpi, ros, medications, allergies, pmh, social, family, additional)
            value: The information provided by the patient
            subcategory: For structured data like HPI or ROS (onset, provocation, quality, etc.)
        """
        try:
            if info_type == "name":
                self.patient_info["name"] = value
            elif info_type == "appointment_date":
                self.patient_info["appointment_date"] = value
            elif info_type == "chief_complaint":
                self.patient_info["chief_complaint"] = value
            elif info_type == "infection":
                # Map generic infection/fever info into ROS constitutional
                self.patient_info["review_of_systems"]["constitutional"] = value
            elif info_type == "hpi" and subcategory:
                # Map common synonyms and ensure we don't drop unknown subcategories
                mapped = subcategory
                if subcategory == "description":
                    mapped = "quality"
                if mapped in self.patient_info["history_of_present_illness"]:
                    self.patient_info["history_of_present_illness"][mapped] = value
                else:
                    # Store unknown HPI data in additional notes so it's not lost
                    existing = self.patient_info.get("additional_notes", "")
                    joiner = "\n" if existing else ""
                    self.patient_info["additional_notes"] = f"{existing}{joiner}HPI ({subcategory}): {value}"
            elif info_type == "name":
                self.patient_info["name"] = value
                self.id_name_done = True
            elif info_type == "chief_complaint":
                self.patient_info["chief_complaint"] = value
                self.id_chief_done = True
            elif info_type == "ros" and subcategory:
                if subcategory in self.patient_info["review_of_systems"]:
                    self.patient_info["review_of_systems"][subcategory] = value
                elif subcategory.lower() == "dyspnea":
                    # Map dyspnea to respiratory ROS
                    self.patient_info["review_of_systems"]["respiratory"] = value
            elif info_type == "pertinent_negative":
                self.patient_info["review_of_systems"]["pertinent_negatives"].append(value)
            elif info_type == "medications":
                self.patient_info["medications"] = value
            elif info_type == "allergies":
                self.patient_info["allergies"] = value
            elif info_type == "pmh" or info_type == "past_medical_history":
                self.patient_info["past_medical_history"] = value
            elif info_type == "social":
                self.patient_info["social_history"] = value
            elif info_type == "family":
                self.patient_info["family_history"] = value
            elif info_type == "additional":
                self.patient_info["additional_notes"] = value
            else:
                # Map some common unknown categories to safe fields
                low = (info_type or "").lower()
                if "pre-visit" in low:
                    # ignore availability/consent flags
                    return {"status": "ignored", "message": "availability noted"}
                if "infection" in low or "fever" in low or "post-op" in low or "postop" in low:
                    self.patient_info["review_of_systems"]["constitutional"] = value
                elif "anesthesia" in low:
                    pmh = self.patient_info.get("past_medical_history", "")
                    joiner = "; " if pmh else ""
                    self.patient_info["past_medical_history"] = f"{pmh}{joiner}Anesthesia: {value}".strip()
                elif "sleep" in low or "apnea" in low:
                    # Map OSA-related items to HPI timing/quality and ROS respiratory
                    existing_ros = self.patient_info["review_of_systems"].get("respiratory", "")
                    tag = (subcategory or "OSA").replace("_", " ")
                    # Build human-friendly fragment without internal tags
                    human_item = value if value and value.lower() not in {"yes", "no"} else tag
                    new_ros = (existing_ros + ("; " if existing_ros and human_item else "") + (human_item or "")).strip()
                    self.patient_info["review_of_systems"]["respiratory"] = new_ros
                    # If daytime sleepiness, also reflect in HPI timing
                    if (subcategory or "").lower().startswith("daytime"):
                        self.patient_info["history_of_present_illness"]["timing"] = "daytime sleepiness present"
                elif "chest" in low and "pain" in low:
                    self.patient_info["history_of_present_illness"]["quality"] = (self.patient_info["history_of_present_illness"].get("quality", "") + ("; " if self.patient_info["history_of_present_illness"].get("quality") else "") + value).strip()
                else:
                    # Fallback: stash in additional notes
                    existing = self.patient_info.get("additional_notes", "")
                    joiner = "\n" if existing else ""
                    self.patient_info["additional_notes"] = f"{existing}{joiner}{info_type}{(' ('+subcategory+')') if subcategory else ''}: {value}"
            
            who = self.participant.identity if self.participant else 'unknown'
            logger.info(f"Recorded {info_type}: {value} for {who}")
            return {
                "status": "info_recorded",
                "info_type": info_type,
                "value": value,
                "all_info": self.patient_info
            }
        except Exception as e:
            logger.error(f"Error recording patient info: {e}")
            return {"status": "error", "message": f"Error recording info: {str(e)}"}

    @function_tool()
    async def summarize_and_confirm(self, ctx: RunContext):
        """Generate comprehensive medical summary and ask patient to confirm"""
        # If summary already delivered, don't repeat speaking it
        if getattr(self, 'summary_done', False):
            return {
                "status": "summary_already_delivered",
                "summary": (self.call_summary or {}).get("medical_report", ""),
                "patient_info": self.patient_info,
                "call_summary": self.call_summary,
                "question_count": self.question_count
            }
        # Generate professional medical report
        medical_report = self.generate_medical_report()

        # Create patient-friendly summary
        if self.doctor_note:
            # Note-driven: short paragraph, no markdown; only include collected fields
            parts = []
            if self.patient_info['chief_complaint']:
                parts.append(f"Primary concern: {self.patient_info['chief_complaint']}.")
            hpi_bits = []
            hpi = self.patient_info['history_of_present_illness']
            if hpi['onset']:
                hpi_bits.append(f"started {hpi['onset']}")
            if hpi['severity']:
                hpi_bits.append(f"severity {hpi['severity']}")
            if hpi['duration']:
                hpi_bits.append(f"duration {hpi['duration']}")
            if hpi['quality']:
                hpi_bits.append(hpi['quality'])
            if hpi['radiation']:
                hpi_bits.append(f"radiation {hpi['radiation']}")
            if hpi_bits:
                parts.append("HPI: " + ", ".join(hpi_bits) + ".")
            if self.patient_info['past_medical_history']:
                parts.append(f"History: {self.patient_info['past_medical_history']}.")
            if self.patient_info['medications']:
                parts.append(f"Meds: {self.patient_info['medications']}.")
            if self.patient_info['allergies']:
                parts.append(f"Allergies: {self.patient_info['allergies']}.")
            body = " ".join(parts) if parts else "I’ve captured your details so far."
            patient_summary = (
                f"Alright—{body} Does that look right, or should I adjust anything?"
            )
        else:
            patient_summary = f"""
        Let me summarize what we've collected for your medical visit:
        
        **Primary Concern:** {self.patient_info['chief_complaint'] or 'Not provided'}
        
        **History of Illness:** {self.patient_info['history_of_present_illness']['onset'] or 'Not specified'}{', ' + self.patient_info['history_of_present_illness']['severity'] if self.patient_info['history_of_present_illness']['severity'] else ''}{', ' + self.patient_info['history_of_present_illness']['duration'] if self.patient_info['history_of_present_illness']['duration'] else ''}
        
        **Medical History:** {self.patient_info['past_medical_history'] or 'None reported'}
        **Current Medications:** {self.patient_info['medications'] or 'None reported'}
        **Allergies:** {self.patient_info['allergies'] or 'None reported'}
        
        **Questions Asked:** {self.question_count} out of {self.max_questions}
        
        Is this information complete and accurate? If yes, I'll end the call. If anything needs correction, please let me know.
        """
        
        logger.info(f"Summarizing medical interview for {self.participant.identity}: {self.question_count} questions asked")
        
        # Store the summary for later retrieval
        self.call_summary = {
            "patient_info": self.patient_info.copy(),
            "call_duration": "in_progress",
            "status": "summary_generated",
            "timestamp": datetime.now().isoformat(),
            "question_count": self.question_count,
            "medical_report": medical_report
        }
        self.ready_to_end = True
        self.awaiting_confirmation = True
        self.user_confirmed = False
        self.summary_done = True

        # Speak the patient-facing summary aloud so the user hears it
        try:
            await ctx.session.say(patient_summary)
            # Brief confirmation prompt
            await ctx.session.say("Does that sound correct, or would you like me to adjust anything?")
        except Exception as e:
            logger.error(f"Error speaking summary: {e}")
        
        return {
            "status": "summary_ready",
            "summary": patient_summary,
            "patient_info": self.patient_info,
            "call_summary": self.call_summary,
            "question_count": self.question_count
        }

    @function_tool()
    async def save_notes(self, ctx: RunContext):
        """Save current medical report to TXT file"""
        try:
            current_summary = {
                "patient_info": self.patient_info.copy(),
                "call_duration": "in_progress",
                "status": "notes_saved",
                "timestamp": datetime.now().isoformat(),
                "question_count": self.question_count
            }
            
            filepath = await self.save_medical_report(current_summary)
            if filepath:
                return {
                    "status": "notes_saved",
                    "filepath": filepath,
                    "message": "Medical report has been saved as TXT file"
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to save medical report"
                }
        except Exception as e:
            logger.error(f"Error in save_notes: {e}")
            return {
                "status": "error",
                "message": f"Error saving notes: {str(e)}"
            }

    @function_tool()
    async def detected_answering_machine(self, ctx: RunContext):
        """Called when the call reaches voicemail. Use this tool AFTER you hear the voicemail greeting"""
        # If not connected to a remote participant yet (e.g., between retries), ignore
        if not self.connected or not self.participant:
            logger.info("voicemail detected during dialing/retry; ignoring until connected")
            return {"status": "ignored_during_retry"}

        try:
            identity = self.participant.identity
            phone_number = identity or self.dial_info.get("phone_number", "unknown")
            logger.info(f"detected answering machine for {phone_number}")

            # Save a voicemail note, then hang up
            save_no_answer_note(phone_number, reason="voicemail", dial_info=self.dial_info)
        finally:
            await self.hangup()

    def update_interview_phase(self, new_phase: str):
        """Update the current interview phase"""
        self.interview_phase = new_phase
        logger.info(f"Interview phase updated to: {new_phase}")

    def increment_question_count(self):
        """Increment the question counter"""
        self.question_count += 1
        if self.question_count >= self.max_questions:
            logger.info(f"Maximum questions ({self.max_questions}) reached")
        return self.question_count

async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect()

    # when dispatching the agent, we'll pass it the approriate info to dial the user
    # dial_info is a dict with the following keys:
    # - phone_number: the phone number to dial
    # - transfer_to: the phone number to transfer the call to when requested
    
    metadata_str = ctx.job.metadata
    dial_info = {}
    logger.info(f"Raw metadata: {metadata_str}")
    
    if metadata_str:
        try:
            # Try to parse as JSON first
            import json
            dial_info = json.loads(metadata_str)
            logger.info(f"Parsed JSON metadata: {dial_info}")
        except json.JSONDecodeError:
            # Fallback to string parsing
            logger.info("JSON parsing failed, trying string parsing")
            clean_str = metadata_str.strip('{}')
            for item in clean_str.split(','):
                if ':' in item:
                    key, value = item.split(':', 1)
                    key = key.strip().strip('"')
                    value = value.strip().strip('"')
                    dial_info[key] = value
            logger.info(f"Parsed string metadata: {dial_info}")
    
    logger.info(f"Final dial_info: {dial_info}")
    
    # Validate required fields
    logger.info(f"Checking for phone_number in dial_info: {dial_info}")
    if not dial_info.get("phone_number"):
        logger.error(f"No phone_number found in metadata. Available keys: {list(dial_info.keys())}")
        logger.error(f"Metadata string was: {metadata_str}")
        ctx.shutdown()
        return
        
    participant_identity = phone_number = dial_info["phone_number"]

    # Initialize the professional medical intake agent
    agent = OutboundCaller(
        name="Patient",  # Generic name since we'll get it from the patient
        appointment_time="upcoming appointment",
        dial_info=dial_info,
    )

    # Determine doctor-note mode early (affects session config)
    doctor_note = (dial_info.get("doctor_note") or "").strip()
    doctor_note_mode = bool(doctor_note)

    # Optimized session configuration for minimal delays
    session = AgentSession(
        vad=silero.VAD.load(
            activation_threshold=0.30,
            min_speech_duration=0.10,
            min_silence_duration=0.35,
            prefix_padding_duration=0.10,
            max_buffered_speech=20.0,
            force_cpu=False,
        ),
        stt=deepgram.STT(
            model="nova-3",
            language="en-US",
            endpointing_ms=150,
        ),
        tts=deepgram.TTS(
            model="aura-asteria-en",
        ),
        llm=google.LLM(
            model="gemini-1.5-flash",  # Supports function calling with higher quotas
            temperature=0.7,  # Slightly lower for more consistent medical interviewing
            api_key=google_api_key,
        ),
        # Disable preemptive generation when following a strict question list
        preemptive_generation=False if doctor_note_mode else True,
    )

    # Add event handlers for better conversation flow
    note_questions: list[str] = []
    note_q_idx: int = 0
    started_questions: bool = False
    fallback_task = None
    waiting_for_consent: bool = False

    def _questions_from_note(note: str) -> list[str]:
        # Simple heuristic conversion of provider note into questions without LLM
        text = (note or "").strip()
        # Sanitize: remove obvious template headings and duplicate words
        for junk in ["PRE-VISIT MEDICAL INTAKE REPORT", "Report Details:"]:
            text = text.replace(junk, " ")
        if not text:
            return []
        # Split by semicolons or periods for clauses
        import re
        parts = [p.strip() for p in re.split(r"[;\n]+|(?<=[.])\s+", text) if p.strip()]
        questions: list[str] = []
        for p in parts:
            low = p.lower()
            if not p:
                continue
            # Map common phrases to concise questions
            if "functional capacity" in low or "climb" in low:
                questions.append("Can you climb two flights of stairs or carry groceries without symptoms?")
            elif "heart" in low or "lung" in low or "kidney" in low:
                questions.append("Do you have a history of heart, lung, or kidney disease?")
            elif "chest pain" in low or "dyspnea" in low or "shortness of breath" in low:
                questions.append("Do you get chest pain or shortness of breath with exertion?")
            elif "sleep apnea" in low or "obstructive" in low:
                questions.append("Have you been told you have obstructive sleep apnea or do you snore loudly and stop breathing during sleep?")
            elif "anesthesia" in low:
                questions.append("Have you had any problems with anesthesia in the past?")
            elif "anticoagulant" in low or "antiplatelet" in low or "sglt2" in low or "current meds" in low or "meds" in low:
                questions.append("What prescription medications are you taking now, including any blood thinners or SGLT2 inhibitors?")
            elif "infection" in low or "fever" in low:
                questions.append("Have you had any recent infections or fevers?")
            elif "allerg" in low:
                questions.append("Do you have any medication or latex allergies?")
            else:
                # Fallback to a short, neutral question to avoid reading the note verbatim
                questions.append("Based on your doctor’s note, what symptoms are you experiencing right now?")
        # Deduplicate and cap to 12
        seen = set()
        final: list[str] = []
        for q in questions:
            if q.lower() not in seen:
                final.append(q)
                seen.add(q.lower())
            if len(final) >= 12:
                break
        return final

    async def _start_note_questions_if_needed():
        nonlocal started_questions, note_q_idx, waiting_for_consent
        if started_questions or not (doctor_note_mode and note_questions):
            return
        started_questions = True
        # Say intro, then wait for explicit user response before asking Q1
        await session.say("Hello, this is your pre-visit intake assistant. I'm here to collect important medical information to help prepare for your appointment. Is now a good time to talk?")
        await asyncio.sleep(0.2)
        waiting_for_consent = True

    @session.on("user_message")
    def on_user_message(message):
        logger.info(f"User said: {message.text}")
        # Meta-questions: always stay in intake scope
        meta = message.text.strip().lower()
        if meta in {"what can you do?", "what can you do for me?", "who are you?", "why did you call me?", "what you can do for me?"}:
            async def _meta_reply():
                await session.say("I’m your pre-visit intake assistant. I’ll ask a few brief health questions to help your provider prepare for your appointment. We’ll keep it quick and focused.")
                # Continue with next medical question if in note-mode
                if doctor_note_mode and note_questions:
                    nonlocal note_q_idx
                    if note_q_idx < len(note_questions):
                        q = note_questions[note_q_idx]
                        note_q_idx += 1
                        agent.increment_question_count()
                        await asyncio.sleep(0.2)
                        await session.say(q)
            asyncio.create_task(_meta_reply())
            return

        # If awaiting confirmation after summary, interpret yes/no explicitly
        if getattr(agent, 'awaiting_confirmation', False):
            yes_set = {"yes", "yep", "yeah", "correct", "that’s correct", "its correct", "it's correct", "ok", "okay", "k"}
            no_set = {"no", "nope", "not correct", "needs changes", "change", "adjust"}
            text_low = meta
            if any(w in text_low for w in yes_set):
                agent.user_confirmed = True
                agent.awaiting_confirmation = False
                async def _end_after_confirm():
                    await session.say("Great, I’ll wrap up now. Thank you and take care.")
                    await session.call_tool("end_call")
                asyncio.create_task(_end_after_confirm())
                return
            if any(w in text_low for w in no_set):
                agent.user_confirmed = False
                agent.awaiting_confirmation = False
                async def _adjust():
                    await session.say("No problem. Tell me what needs to be updated, and I’ll correct it.")
                asyncio.create_task(_adjust())
                return
        # If doctor-note mode and not started, begin Q list upon first utterance
        if doctor_note_mode and note_questions and not started_questions:
            async def _kickoff():
                # Do not count this as a question response
                await _start_note_questions_if_needed()
            asyncio.create_task(_kickoff())
            return
        # If waiting for consent after intro in doctor-note mode, run identification first
        if doctor_note_mode:
            nonlocal waiting_for_consent, note_q_idx
            if started_questions and waiting_for_consent:
                waiting_for_consent = False
                async def _ident_then_q1():
                    # Step 1: confirm name only
                    if not agent.id_name_done:
                        await session.say("Great—may I confirm your full name, please?")
                        agent.id_name_done = True
                        return
                    # Then proceed to first note-derived question
                    if note_q_idx == 0 and note_questions:
                        q = note_questions[0]
                        note_q_idx = 1
                        agent.increment_question_count()
                        agent.last_question = q
                        await asyncio.sleep(0.2)
                        await session.say(q)
                asyncio.create_task(_ident_then_q1())
                return
        # Increment question count for each user response (after start)
        agent.increment_question_count()
        
        async def ask_next():
            # If planning, ignore tool-freeform generation and do nothing
            if agent.planning_mode:
                    return
            # Do not ask new questions if we're awaiting confirmation or summary already done
            if getattr(agent, 'awaiting_confirmation', False) or getattr(agent, 'summary_done', False):
                return
            if doctor_note_mode and note_questions:
                nonlocal note_q_idx
                # If identification not complete, finish it before continuing (name only in note-mode)
                if not agent.id_name_done:
                    await session.say("Could you please confirm your full name?")
                    agent.id_name_done = True
                    return
                if note_q_idx < len(note_questions):
                    q = note_questions[note_q_idx]
                    note_q_idx += 1
                    agent.increment_question_count()
                    await asyncio.sleep(0.3)
                    agent.last_question = q
                    await session.say(q)
                    return
                # Finished doctor note questions -> summarize and end
                # Only summarize when enough info has been collected
                if agent._count_recorded_items() >= 3:
                    await session.call_tool("summarize_and_confirm")
                else:
                    # Ask one final generic wrap-up question before summarizing
                    await session.say("Before I summarize, is there anything else important you want your provider to know?")
                # Do not auto-end; allow the patient to confirm or ask to end
                return
            # Normal flow fallback
            await session.generate_reply(instructions=f"Respond to: '{message.text}' with the next medically appropriate single question.")

        asyncio.create_task(ask_next())
    
    # Add error handling for rate limits and other errors
    @session.on("error")
    def on_error(error):
        if "429" in str(error) or "quota" in str(error).lower():
            logger.warning("Rate limit hit, waiting before retry...")
            # The agent will automatically retry after a delay
        elif "function calling is not enabled" in str(error).lower():
            logger.error("Model doesn't support function calling - this will break the agent")
        else:
            logger.error(f"Session error: {error}")
    
    # Add connection status monitoring
    @session.on("connected")
    def on_connected():
        logger.info("Agent session connected successfully")
    
    @session.on("disconnected")
    def on_disconnected():
        logger.info("Agent session disconnected")

    # start the session first before dialing, to ensure that when the user picks up
    # the agent does not miss anything the user says
    session_started = asyncio.create_task(
        session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                # enable Krisp background voice and noise removal
                noise_cancellation=noise_cancellation.BVCTelephony(),
            ),
        )
    )
    # `create_sip_participant` starts dialing the user with retry on timeout
    dial_success = False
    total_attempts = max(1, outbound_retry_count + 1)
    for attempt_idx in range(1, total_attempts + 1):
        try:
            logger.info(
                f"Dial attempt {attempt_idx}/{total_attempts} to {phone_number} with timeout {outbound_call_timeout_s}s"
            )
            await asyncio.wait_for(
                ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=outbound_trunk_id,
                sip_call_to=phone_number,
                participant_identity=participant_identity,
                # function blocks until user answers the call, or if the call fails
                wait_until_answered=True,
            )
                ),
                timeout=outbound_call_timeout_s,
            )
            dial_success = True
            break
        except asyncio.TimeoutError:
            logger.warning(
                f"Attempt {attempt_idx} timed out after {outbound_call_timeout_s}s (no answer)"
            )
            if attempt_idx < total_attempts:
                logger.info(f"Retrying in {outbound_retry_delay_s}s...")
                await asyncio.sleep(outbound_retry_delay_s)
                continue
            # Final failure after retries
            save_no_answer_note(phone_number, reason="timeout", dial_info=dial_info)
            ctx.shutdown()
            return
        except api.TwirpError as e:
            logger.error(
                f"error creating SIP participant: {e.message}, "
                f"SIP status: {e.metadata.get('sip_status_code')} "
                f"{e.metadata.get('sip_status')}"
            )
            reason = f"SIP error {e.metadata.get('sip_status_code')} {e.metadata.get('sip_status')}"
            save_no_answer_note(phone_number, reason=reason, dial_info=dial_info)
            ctx.shutdown()
            return

    if not dial_success:
        return

    # wait for the agent session start and participant join
    await session_started
    participant = await ctx.wait_for_participant(identity=participant_identity)
    logger.info(f"participant joined: {participant.identity}")

    agent.set_participant(participant)

    # Egress/recording removed
    egress_id = None

    if doctor_note_mode:
        logger.info("Starting doctor-note-driven interview")
        # Build question plan locally without invoking LLM audio/tools
        agent.planning_mode = True
        try:
            note_questions = _questions_from_note(doctor_note)
        finally:
            agent.planning_mode = False
        if note_questions:
            # Wait for first user utterance, or fallback after 2s
            async def _fallback_start():
                await asyncio.sleep(2.0)
                await _start_note_questions_if_needed()
            fallback_task = asyncio.create_task(_fallback_start())
        else:
            # Fallback to normal opening if planning failed
            logger.info("Doctor note plan empty; falling back to normal intro")
            await session.generate_reply(
                instructions="""
                Say briefly and professionally:
                "Hello, this is your pre-visit intake assistant. I'm here to collect important medical information to help prepare for your appointment. Is now a good time to talk?"
                Keep it professional, friendly, and under 20 seconds.
                """
            )
    else:
        # Start the conversation (normal)
        logger.info("Starting professional medical intake interview")
        await session.generate_reply(
                instructions="""
                Say briefly and professionally:
                "Hello, this is your pre-visit intake assistant. I'm here to collect important medical information to help prepare for your appointment. Is now a good time to talk?"
                Keep it professional, friendly, and under 20 seconds.
                """
        )
    # Register a best-effort stop on disconnect
    @session.on("disconnected")
    def _on_disc():
        # No egress to stop anymore
        return


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="outbound-caller",
        )
    )