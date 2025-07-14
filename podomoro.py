import streamlit as st
from openai import OpenAI
from typing import Dict, List
import time

# Configure the page
st.set_page_config(
    page_title="AI-Powered Quiz App",
    page_icon="🧠",
    layout="wide"
)

# Initialize OpenAI client (you'll need to set your API key)
# You can set this in Streamlit secrets or as an environment variable
# if "openai_api_key" not in st.session_state:
st.session_state.openai_api_key = st.secrets["OPENAI_API_KEY"]

if "openai_client" not in st.session_state:
    st.session_state.openai_client = None

if not st.session_state.openai_api_key:
    api_key = st.text_input("Enter your OpenAI API Key:", type="password", key="api_key_input")
    if api_key:
        st.session_state.openai_api_key = api_key
        st.session_state.openai_client = OpenAI(api_key=api_key)
    else:
        st.warning("Please enter your OpenAI API key to continue.")
        st.stop()
else:
    if st.session_state.openai_client is None:
        st.session_state.openai_client = OpenAI(api_key=st.session_state.openai_api_key)

# Sample quiz questions - you can modify these
# QUIZ_QUESTIONS = [
#     {
#         "question": "What is the capital of France?",
#         "options": ["London", "Berlin", "Paris", "Madrid"],
#         "correct": 2,
#         "topic": "Geography"
#     },
#     {
#         "question": "Which planet is known as the Red Planet?",
#         "options": ["Venus", "Mars", "Jupiter", "Saturn"],
#         "correct": 1,
#         "topic": "Astronomy"
#     },
#     {
#         "question": "What is 15 × 8?",
#         "options": ["120", "125", "115", "130"],
#         "correct": 0,
#         "topic": "Mathematics"
#     },
#     {
#         "question": "Who wrote 'Romeo and Juliet'?",
#         "options": ["Charles Dickens", "William Shakespeare", "Jane Austen", "Mark Twain"],
#         "correct": 1,
#         "topic": "Literature"
#     },
#     {
#         "question": "What is the chemical symbol for gold?",
#         "options": ["Go", "Gd", "Au", "Ag"],
#         "correct": 2,
#         "topic": "Chemistry"
#     }
# ]


import pandas as pd

def transform_excel_to_quiz_questions(file_path: str):
    df = pd.read_excel(file_path)

    # Buat kolom 'correct_index' kalau belum ada
    if 'correct_index' not in df.columns:
        correct_mapping = {
            "option_a": 0,
            "option_b": 1,
            "option_c": 2,
            "option_d": 3,
            "option_e": 4,  # kalau ada
        }
        df["correct_index"] = df["correct"].map(correct_mapping)

    quiz_questions = []
    for _, row in df.iterrows():
        options = [row['option_a'], row['option_b'], row['option_c'], row['option_d']]
        if 'option_e' in row:  # optional
            options.append(row['option_e'])

        question_data = {
            "question": row["question"],
            "options": options,
            "correct": int(row["correct_index"]),
            "topic": row["topic"]
        }
        quiz_questions.append(question_data)

    return quiz_questions


QUIZ_QUESTIONS = transform_excel_to_quiz_questions("data/quiz_questions.xlsx")

def get_ai_explanation(question: str, correct_answer: str, user_answer: str, topic: str, other_option : List) -> str:
    """Get AI explanation for wrong answers"""
    try:
        client = st.session_state.openai_client
        
        prompt = f"""
        A student answered a quiz question incorrectly. Please provide a helpful, encouraging explanation.
        
        Topic: {topic}
        Question: {question}
        Correct Answer: {correct_answer}
        Student's Answer: {user_answer} 
        all the other options: {other_option}
        
        Please provide:
        1. A brief explanation of why the correct answer is right
        2. Why the student's answer might be incorrect
        3. A helpful tip to remember this for next time
        4. explain other options as a clarification and add why they are not correct
        
        Keep the tone encouraging and educational. Limit to 3-4 sentences.
        """
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"I'd be happy to help explain this! The correct answer is {correct_answer}. Consider reviewing {topic} concepts to strengthen your understanding. (Error: {str(e)})"

def initialize_session_state():
    """Initialize session state variables"""
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'score' not in st.session_state:
        st.session_state.score = 0
    if 'answers' not in st.session_state:
        st.session_state.answers = []
    if 'quiz_complete' not in st.session_state:
        st.session_state.quiz_complete = False
    if 'question_answered' not in st.session_state:
        st.session_state.question_answered = False
    if 'show_next_button' not in st.session_state:
        st.session_state.show_next_button = False
    if 'ai_explanation' not in st.session_state:
        st.session_state.ai_explanation = ""

def reset_quiz():
    """Reset the quiz to start over"""
    st.session_state.current_question = 0
    st.session_state.score = 0
    st.session_state.answers = []
    st.session_state.quiz_complete = False
    st.session_state.question_answered = False
    st.session_state.show_next_button = False
    st.session_state.ai_explanation = ""

def main():
    st.title("🧠 AI-Powered Quiz Application")
    st.markdown("Test your knowledge and get AI-powered explanations for wrong answers!")
    
    initialize_session_state()
    
    # Sidebar with progress and score
    with st.sidebar:
        st.header("Quiz Progress")
        if not st.session_state.quiz_complete:
            progress = st.session_state.current_question / len(QUIZ_QUESTIONS)
            st.progress(progress)
            st.write(f"Question {st.session_state.current_question + 1} of {len(QUIZ_QUESTIONS)}")
        
        st.write(f"**Current Score:** {st.session_state.score}/{len(QUIZ_QUESTIONS)}")
        
        if st.button("Reset Quiz", key="reset_quiz"):
            reset_quiz()
    
    # Main quiz interface
    if not st.session_state.quiz_complete:
        current_q = QUIZ_QUESTIONS[st.session_state.current_question]
        
        # Display question
        st.subheader(f"Question {st.session_state.current_question + 1}")
        st.write(f"**Topic:** {current_q['topic']}")
        st.write(current_q['question'])
        
        # Question form
        if not st.session_state.question_answered:
            with st.form(key=f"question_form_{st.session_state.current_question}"):
                user_answer = st.radio(
                    "Choose your answer:",
                    range(len(current_q['options'])),
                    format_func=lambda x: current_q['options'][x],
                    key=f"answer_{st.session_state.current_question}"
                )
                
                submit_button = st.form_submit_button("Submit Answer", type="primary")
                
                if submit_button:
                    is_correct = user_answer == current_q['correct']
                    
                    # Store the answer
                    answer_data = {
                        'question': current_q['question'],
                        'user_answer': current_q['options'][user_answer],
                        'correct_answer': current_q['options'][current_q['correct']],
                        'is_correct': is_correct,
                        'topic': current_q['topic'], 
                        'options': [items for items in current_q['options']]
                    }
                    st.session_state.answers.append(answer_data)
                    
                    if is_correct:
                        st.session_state.score += 1
                        st.success("✅ Correct! Well done!")
                    else:
                        st.error("❌ Incorrect. Let me get an AI explanation for you...")
                        # Get AI explanation immediately
                        with st.spinner("Getting AI explanation..."):
                            explanation = get_ai_explanation(
                                current_q['question'],
                                current_q['options'][current_q['correct']],
                                current_q['options'][user_answer],
                                current_q['topic'] ,
                                current_q['options']
                            )
                        st.session_state.ai_explanation = explanation
                        st.info(f"🤖 **AI Explanation:** {explanation}")
                    
                    st.session_state.question_answered = True
                    st.session_state.show_next_button = True
        
        # Show next button or complete quiz
        if st.session_state.show_next_button:
            if st.session_state.current_question < len(QUIZ_QUESTIONS) - 1:
                if st.button("Next Question", type="primary", key="next_question"):
                    st.session_state.current_question += 1
                    st.session_state.question_answered = False
                    st.session_state.show_next_button = False
                    st.session_state.ai_explanation = ""
            else:
                if st.button("Complete Quiz", type="primary", key="complete_quiz"):
                    st.session_state.quiz_complete = True
                    st.session_state.question_answered = False
                    st.session_state.show_next_button = False
    
    else:
        # Quiz complete - show results
        st.balloons()
        st.success("🎉 Quiz Complete!")
        
        # Calculate percentage
        percentage = (st.session_state.score / len(QUIZ_QUESTIONS)) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Score", f"{st.session_state.score}/{len(QUIZ_QUESTIONS)}")
        with col2:
            st.metric("Percentage", f"{percentage:.1f}%")
        with col3:
            if percentage >= 80:
                st.metric("Grade", "A", delta="Excellent!")
            elif percentage >= 60:
                st.metric("Grade", "B", delta="Good job!")
            else:
                st.metric("Grade", "C", delta="Keep practicing!")
        
        # Detailed results
        st.subheader("📊 Detailed Results")
        
        for i, answer in enumerate(st.session_state.answers):
            with st.expander(f"Question {i+1}: {answer['question'][:50]}..."):
                st.write(f"**Question:** {answer['question']}")
                st.write(f"**Your Answer:** {answer['user_answer']}")
                st.write(f"**Correct Answer:** {answer['correct_answer']}")
                
                if answer['is_correct']:
                    st.success("✅ Correct!")
                else:
                    st.error("❌ Incorrect")
                    
                    # Button to get AI explanation for review
                    if st.button(f"Get AI Explanation", key=f"explain_{i}"):
                        with st.spinner("Getting explanation..."):
                            explanation = get_ai_explanation(
                                answer['question'],
                                answer['correct_answer'],
                                answer['user_answer'],
                                answer['topic'], 
                                answer['options']
                            )
                        st.markdown(f" **AI Explanation:** {explanation}")
        
        # Restart option
        if st.button("Take Quiz Again", type="primary", key="restart_quiz"):
            reset_quiz()

if __name__ == "__main__":
    main()