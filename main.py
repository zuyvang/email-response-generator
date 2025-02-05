# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import threading
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch
from transformers import pipeline
from textblob import TextBlob
import openai
import re



@dataclass
class EmailSentiment:
    polarity: float
    subjectivity: float
    overall_sentiment: str


@dataclass
class EmailEmotion:
    label: str
    confidence: float


@dataclass
class Relationship:
    sender: str
    receiver: str


class EmailAnalyzer:
    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        print(f"Using {'GPU' if device == 0 else 'CPU'} for emotion analysis")

        try:
            self.emotion_analyzer = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=device
            )
            print("Emotion analyzer initialized successfully")
        except Exception as e:
            print(f"Error initializing emotion analyzer: {str(e)}")
            self.emotion_analyzer = None

    def detect_emotions(self, text: str) -> List[EmailEmotion]:
        try:
            results = self.emotion_analyzer(text)
            sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
            return [
                EmailEmotion(label=result['label'], confidence=result['score'])
                for result in sorted_results
            ]
        except Exception as e:
            print(f"Error detecting emotions: {str(e)}")
            return [EmailEmotion(label="neutral", confidence=1.0)]

    def analyze_sentiment(self, text: str) -> EmailSentiment:
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment

            if sentiment.polarity > 0:
                overall = "Positive"
            elif sentiment.polarity < 0:
                overall = "Negative"
            else:
                overall = "Neutral"

            return EmailSentiment(
                polarity=sentiment.polarity,
                subjectivity=sentiment.subjectivity,
                overall_sentiment=overall
            )
        except Exception as e:
            print(f"Error analyzing sentiment: {str(e)}")
            return EmailSentiment(0.0, 0.0, "Error")

    def identify_relationship(self, email_text: str) -> Relationship:
        try:
            messages = [{
                "role": "user",
                "content": f"""Identify the relationship between the sender and receiver in this email:
                {email_text}
                Provide the answer in this format:
                SENDER: <relationship description>
                RECEIVER: <relationship description>"""
            }]

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=200
            )

            content = response['choices'][0]['message']['content']
            pattern = r"SENDER:\s*(.+?)\s*\nRECEIVER:\s*(.+)"
            match = re.search(pattern, content)

            if not match:
                return Relationship("Unknown", "Unknown")

            return Relationship(
                sender=match.group(1).strip(),
                receiver=match.group(2).strip()
            )

        except Exception as e:
            print(f"Error identifying relationship: {str(e)}")
            return Relationship("Unknown", "Unknown")

    def generate_reply(self,
                       received_email: str,
                       relationship: Relationship,
                       tone: str = "professional",
                       style: str = "detailed") -> str:
        try:
            system_prompt = f"""You are a {relationship.receiver} responding to a {relationship.sender}.
            Your response should be in a {tone} tone and {style} style.

            Style Guidelines:
            - Detailed: Include comprehensive information and thorough explanations
            - Concise: Keep it brief but complete
            - Bullet Points: Use bullet points for key information
            - Formal: Use formal business language
            - Step-by-Step: Break down information into clear steps
            - Question-Answer: Structure as Q&A format
            - Summary First: Start with key points then details

            Tone Guidelines:
            - Match formality to the original email
            - Use appropriate greetings and sign-offs
            - If casual/informal: Use contractions, simpler language
            - If formal: Use proper business email format
            - If confrontational/stern: Be firm but professional
            - If humorous: Include appropriate light humor
            - If passive-aggressive: Use subtle implications

            Remember to:
            - Include specific details from the original email
            - Use clear paragraph structure
            - Add appropriate transitional phrases
            - Keep consistent voice throughout"""

            user_prompt = f"""Original Email:
            {received_email}

            Please write a {style} response in a {tone} tone that:
            1. Addresses all points and questions
            2. Uses appropriate greeting and sign-off
            3. Maintains consistent tone throughout
            4. Follows the style guidelines
            5. Includes specific details from the original email
            6. Has clear structure
            7. Adds value to the conversation

            Format as a complete email with proper spacing."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500,
                temperature=0.7,
                presence_penalty=0.6,
                frequency_penalty=0.6
            )

            return response['choices'][0]['message']['content']

        except Exception as e:
            print(f"Error generating reply: {str(e)}")
            return "Error generating reply. Please try again later."


class ModernEmailAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Email Response Generator")
        self.analyzer = EmailAnalyzer()

        self.colors = {
            'bg': '#1e1e1e',
            'secondary_bg': '#2d2d2d',
            'text': '#ffffff',
            'accent': '#007acc',
            'text_area_bg': '#2d2d2d',
            'text_area_fg': '#e0e0e0',
            'button_bg': '#3d3d3d',
            'button_hover': '#4d4d4d'
        }

        self.root.configure(bg=self.colors['bg'])
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self.setup_gui()

    def create_modern_textbox(self, parent, height=10):
        text_widget = scrolledtext.ScrolledText(
            parent,
            height=height,
            font=('Segoe UI', 10),
            bg=self.colors['text_area_bg'],
            fg=self.colors['text_area_fg'],
            wrap=tk.WORD,
            borderwidth=1,
            relief="solid",
            insertbackground=self.colors['text_area_fg']
        )
        return text_widget

    def setup_gui(self):
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

        main_frame.grid_columnconfigure(0, weight=1)
        for i in range(12):
            main_frame.grid_rowconfigure(i, weight=1)

        # Title
        title_label = tk.Label(
            main_frame,
            text="Email Analysis & Suggested Response Tool",
            font=('Segoe UI', 14, 'bold'),
            bg=self.colors['bg'],
            fg=self.colors['text']
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20), sticky="nw")

        # Input Email Label
        input_label = tk.Label(
            main_frame,
            text="Input Email:",
            font=('Segoe UI', 10),
            bg=self.colors['bg'],
            fg=self.colors['text']
        )
        input_label.grid(row=1, column=0, sticky="w", pady=(0, 5))

        # Input Email Textbox
        self.email_input = self.create_modern_textbox(main_frame)
        self.email_input.grid(row=2, column=0, sticky="nsew", pady=(0, 15))

        # Response Options Label
        response_options_label = tk.Label(
            main_frame,
            text="Response Options:",
            font=('Segoe UI', 10,),
            bg=self.colors['bg'],
            fg=self.colors['text']
        )
        response_options_label.grid(row=3, column=0, sticky="w", pady=(5, 5))

        # Options Frame for Tones and Styles
        options_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        options_frame.grid(row=4, column=0, sticky="ew", pady=(0, 15))
        options_frame.grid_columnconfigure(1, weight=1)
        options_frame.grid_columnconfigure(3, weight=1)

        # Tone 1 Dropdown
        tk.Label(options_frame, text="Tone 1:", bg=self.colors['bg'], fg=self.colors['text'],
                 font=('Segoe UI', 10)).grid(row=0, column=0, padx=(0, 10))

        self.tone1 = ttk.Combobox(
            options_frame,
            values=[
                "Professional", "Friendly", "Empathetic", "Formal", "Direct",
                "Casual", "Assertive", "Stern", "Sarcastic", "Humorous",
                "Blunt", "No-nonsense", "Passive-aggressive", "Apologetic",
                "Enthusiastic", "Cold", "Urgent", "Diplomatic", "Confrontational"
            ],
            font=('Segoe UI', 10)
        )
        self.tone1.set("Professional")
        self.tone1.grid(row=0, column=1, sticky="ew", padx=5)

        # Style 1 Dropdown
        tk.Label(options_frame, text="Style 1:", bg=self.colors['bg'], fg=self.colors['text'],
                 font=('Segoe UI', 10)).grid(row=0, column=2, padx=(20, 10))

        self.style1 = ttk.Combobox(
            options_frame,
            values=[
                "Detailed", "Concise", "Bullet Points", "Formal",
                "Step-by-Step", "Question-Answer", "Summary First"
            ],
            font=('Segoe UI', 10)
        )
        self.style1.set("Detailed")
        self.style1.grid(row=0, column=3, sticky="ew", padx=5)

        # Tone 2 Dropdown
        tk.Label(options_frame, text="Tone 2:", bg=self.colors['bg'], fg=self.colors['text'],
                 font=('Segoe UI', 10)).grid(row=1, column=0, padx=(0, 10), pady=(10, 0))

        self.tone2 = ttk.Combobox(
            options_frame,
            values=[
                "Professional", "Friendly", "Empathetic", "Formal", "Direct",
                "Casual", "Assertive", "Stern", "Sarcastic", "Humorous",
                "Blunt", "No-nonsense", "Passive-aggressive", "Apologetic",
                "Enthusiastic", "Cold", "Urgent", "Diplomatic", "Confrontational"
            ],
            font=('Segoe UI', 10)
        )
        self.tone2.set("Friendly")
        self.tone2.grid(row=1, column=1, sticky="ew", padx=5, pady=(10, 0))

        # Style 2 Dropdown
        tk.Label(options_frame, text="Style 2:", bg=self.colors['bg'], fg=self.colors['text'],
                 font=('Segoe UI', 10)).grid(row=1, column=2, padx=(20, 10), pady=(10, 0))

        self.style2 = ttk.Combobox(
            options_frame,
            values=[
                "Detailed", "Concise", "Bullet Points", "Formal",
                "Step-by-Step", "Question-Answer", "Summary First"
            ],
            font=('Segoe UI', 10)
        )
        self.style2.set("Concise")
        self.style2.grid(row=1, column=3, sticky="ew", padx=5, pady=(10, 0))

        # Analyze Button
        self.analyze_button = tk.Button(
            main_frame,
            text="Analyze & Generate Responses",
            command=self.analyze_email,
            font=('Segoe UI', 10),
            bg=self.colors['accent'],
            fg=self.colors['text'],
            borderwidth=0,
            padx=20,
            pady=10,
            cursor="hand2"
        )
        self.analyze_button.grid(row=5, column=0, pady=20, sticky="ew")

        self.analyze_button.bind('<Enter>', lambda e: e.widget.configure(bg=self.colors['button_hover']))
        self.analyze_button.bind('<Leave>', lambda e: e.widget.configure(bg=self.colors['accent']))

        # Analysis Results
        tk.Label(main_frame, text="Analysis Results:", bg=self.colors['bg'], fg=self.colors['text'],
                 font=('Segoe UI', 10)).grid(row=6, column=0, sticky="w", pady=(0, 5))
        self.analysis_results = self.create_modern_textbox(main_frame, height=4)
        self.analysis_results.grid(row=7, column=0, sticky="nsew", pady=(0, 15))

        # Suggested Response 1
        tk.Label(main_frame, text="Suggested Response 1:", bg=self.colors['bg'], fg=self.colors['text'],
                 font=('Segoe UI', 10)).grid(row=8, column=0, sticky="w", pady=(0, 5))
        self.response1 = self.create_modern_textbox(main_frame, height=6)
        self.response1.grid(row=9, column=0, sticky="nsew", pady=(0, 15))

        # Suggested Response 2
        tk.Label(main_frame, text="Suggested Response 2:", bg=self.colors['bg'], fg=self.colors['text'],
                 font=('Segoe UI', 10)).grid(row=10, column=0, sticky="w", pady=(0, 5))
        self.response2 = self.create_modern_textbox(main_frame, height=6)
        self.response2.grid(row=11, column=0, sticky="nsew", pady=(0, 15))

        # Progress Label
        self.progress_var = tk.StringVar()
        self.progress_label = tk.Label(
            main_frame,
            textvariable=self.progress_var,
            bg=self.colors['bg'],
            fg=self.colors['text'],
            font=('Segoe UI', 10)
        )
        self.progress_label.grid(row=12, column=0, sticky="ew", pady=(0, 10))

    def analyze_email(self):
        self.analyze_button.config(state='disabled')
        self.progress_var.set("Analyzing...")

        # Clear previous results
        self.analysis_results.delete(1.0, tk.END)
        self.response1.delete(1.0, tk.END)
        self.response2.delete(1.0, tk.END)

        threading.Thread(target=self.process_email, daemon=True).start()

    def process_email(self):
        try:
            email_text = self.email_input.get(1.0, tk.END).strip()

            self.progress_var.set("Analyzing email content...")
            emotions = self.analyzer.detect_emotions(email_text)
            sentiment = self.analyzer.analyze_sentiment(email_text)

            self.progress_var.set("Identifying relationships...")
            relationship = self.analyzer.identify_relationship(email_text)

            analysis_text = f"Emotions: {', '.join(f'{e.label} ({e.confidence:.2f})' for e in emotions)}\n"
            analysis_text += f"Sentiment: {sentiment.overall_sentiment} (Polarity: {sentiment.polarity:.2f})\n"
            analysis_text += f"Relationship - From: {relationship.sender}, To: {relationship.receiver}"

            self.analysis_results.insert(tk.END, analysis_text)

            # Generate first response
            self.progress_var.set("Generating first response...")
            response1 = self.analyzer.generate_reply(
                email_text,
                relationship,
                self.tone1.get().lower(),
                self.style1.get().lower()
            )

            # Generate second response
            self.progress_var.set("Generating second response...")
            response2 = self.analyzer.generate_reply(
                email_text,
                relationship,
                self.tone2.get().lower(),
                self.style2.get().lower()
            )

            # Update response text boxes
            self.response1.insert(tk.END, response1)
            self.response2.insert(tk.END, response2)

            self.progress_var.set("Analysis complete!")

        except Exception as e:
            self.progress_var.set(f"Error: {str(e)}")
        finally:
            self.analyze_button.config(state='normal')

# Remove the indentation - these should be at the root level of the file
def main():
    root = tk.Tk()
    root.geometry("800x1000")
    root.minsize(600, 800)
    app = ModernEmailAnalyzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()