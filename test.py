
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions
)

load_dotenv()

# Input configuration
source = "https://www.youtube.com/watch?v=zokMWPsToL0"
language = "hinglish"  # Change to "english" to test Whisper


# Step 1: Process audio
print("\nProcessing audio...")
chunks = process_input(source)

# Step 2: Transcribe audio
print("\nTranscribing audio...")
transcript = transcribe_all(chunks, language=language)

print("\n======== TRANSCRIPT ========\n")
print(transcript)


# Step 3: Generate meeting title
print("\nGenerating meeting title...")
title = generate_title(transcript)

print("\n======== MEETING TITLE ========\n")
print(title)


# Step 4: Summarize meeting
print("\nGenerating meeting summary...")
summary = summarize(transcript)

print("\n======== MEETING SUMMARY ========\n")
print(summary)


# Step 5: Extract action items
print("\nExtracting action items...")
action_items = extract_action_items(transcript)

print("\n======== ACTION ITEMS ========\n")
print(action_items)


# Step 6: Extract key decisions
print("\nExtracting key decisions...")
decisions = extract_key_decisions(transcript)

print("\n======== KEY DECISIONS ========\n")
print(decisions)


# Step 7: Extract unresolved questions
print("\nExtracting questions...")
questions = extract_questions(transcript)

print("\n======== OPEN QUESTIONS ========\n")
print(questions)


print("\n======== PROCESSING COMPLETE ========\n")
