import spacy
import PyPDF2
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ''
    return text

# Function to extract key info from resume
def extract_resume_info(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    skills = []
    experience = []
    education = []
    
    # Extract skills (keywords-based matching)
    skill_keywords = {"Python", "Java", "C++", "Machine Learning", "AI", "NLP", "SQL", "Flask", "Django"}
    for token in doc:
        if token.text in skill_keywords:
            skills.append(token.text)

    # Extract experience
    exp_pattern = re.compile(r'\b(\d{1,2})\s?(years|months)\b', re.IGNORECASE)
    exp_matches = exp_pattern.findall(text)
    for match in exp_matches:
        experience.append(f"{match[0]} {match[1]}")

    # Extract education (degree matching)
    edu_keywords = {"B.Tech", "M.Tech", "BSc", "MSc", "Bachelor", "Master", "PhD", "BE", "ME"}
    for ent in doc.ents:
        if ent.label_ == "ORG" or ent.text in edu_keywords:
            education.append(ent.text)

    return {
        "Skills": list(set(skills)),
        "Experience": experience,
        "Education": list(set(education))
    }

# Function to match resume with job description
def match_resume_with_job(resume_text, job_desc):
    vectorizer = CountVectorizer().fit_transform([resume_text, job_desc])
    similarity = cosine_similarity(vectorizer)[0][1] * 100
    return f"{similarity:.2f}% match"

# Main function
def main():
    pdf_path = "resume.pdf"  # Replace with your resume file
    job_desc = """
    Looking for a Python Developer with experience in AI, ML, and SQL. 
    Proficiency in Flask or Django is a plus.
    """

    print("Extracting Resume...")
    resume_text = extract_text_from_pdf(pdf_path)

    print("\nAnalyzing Resume...")
    info = extract_resume_info(resume_text)
    print("\nResume Information:")
    print(f"Skills: {', '.join(info['Skills'])}")
    print(f"Experience: {', '.join(info['Experience'])}")
    print(f"Education: {', '.join(info['Education'])}")

    print("\nMatching with Job Description...")
    match_score = match_resume_with_job(resume_text, job_desc)
    print(f"\nMatch Score: {match_score}")

if __name__ == "__main__":
    main()

