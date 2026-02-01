# Outbound Caller - AI Medical Intake System

A professional AI-powered medical intake system that conducts automated pre-visit screening calls and generates comprehensive medical reports. Built with LiveKit Agents, Deepgram STT/TTS, and Google Gemini LLM.

## 🏥 Features

### **Professional Medical Interview Protocol**
- **Structured 4-Section Reports**: Primary concern, HPI, medical history, medications
- **Maximum 20 Questions**: Efficient, focused medical interviews
- **OPQRST Framework**: Systematic symptom assessment (Onset, Provocation, Quality, Radiation, Severity, Time)
- **Medical Terminology**: Professional healthcare language
- **Pertinent Negatives**: Captures what patients DON'T have

### **Advanced AI Capabilities**
- **Real-time Speech Recognition**: Deepgram STT with Nova-3 model
- **Natural Voice Generation**: Deepgram TTS with Aura-Asteria model
- **Medical Reasoning**: Google Gemini 1.5 Flash for clinical logic
- **Voice Activity Detection**: Optimized Silero VAD for minimal delays
- **Background Noise Cancellation**: Professional telephony-grade audio processing

### **Professional Report Generation**
- **Clean TXT Format**: No markdown, healthcare provider friendly
- **Structured Data**: Organized by medical categories
- **Call Details**: Complete call metadata and timestamps
- **Patient Summaries**: Concise information for quick review
- **EHR Ready**: Easy integration with electronic health records

## 🚀 Quick Start

### **Prerequisites**
- Python 3.8+
- LiveKit account and credentials
- Deepgram API key
- Google Gemini API key
- SIP trunk for outbound calling (optional)

### **Installation**

1. **Clone the repository**
   ```bash
   git clone https://github.com/AliMusawir/Outbound_Caller-.git
   cd Outbound_Caller-
