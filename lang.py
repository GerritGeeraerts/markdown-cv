import os
import time
from typing import List
import re

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_together import Together
from pydantic import BaseModel, Field

from cv2 import d

load_dotenv()

class ProjectScore(BaseModel):
    project_id: str = Field(description="The project id of the project")
    score: float = Field(description="The score of the project")

class ProjectScores(BaseModel):
    projects: List[ProjectScore] = Field(description="List of project scores")


def replace_think_tags(content, replacement=""):
    """
    Replaces everything between </think> tags, including the tags themselves, with a replacement.

    Args:
        content (str): The input string containing the </think> tags.
        replacement (str): The string to replace the tags and their content with. Defaults to an empty string.

    Returns:
        str: The updated string with the tags and their content replaced.
    """
    # Define the regex pattern to match everything between </think> and </think>, including the tags
    pattern = r"</think>.*?</think>"

    # Use re.DOTALL to ensure the regex captures newlines as well
    updated_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    return updated_content

job_description = """
Functieomschrijving

Our client is a prominent player in the financial sector, recognized for its comprehensive range of banking and financial services. With a global presence, it serves a diverse clientele, including individuals, corporations, and institutions. Emphasizing innovation and customer-centricity, it leverages advanced technologies to deliver tailored solutions that meet the evolving needs of its stakeholders.

What you will be doing
The Tribe AI develops and builds augmented intelligence applications. In this context we are looking for professionals with knowledge of Python and more specifically Django to help design, build and test interactive tools to annotate data and validate results of Machine Learning models.
As a member of the agile team, the dev engineer applies several useful skills (analysis / design / development / testing / integration), which in combination with the skills of the other members of the team compose all required skills to deliver value at the end of each sprint.
More specifically of the Django Dev Engineer we expect knowledge of the Django templating system, htmx, usage of background job queues and connection to other API.

Function description
As part of the team, the dev engineer commits to help the team to deliver value at the end of each sprint. This is achieved by interacting with out stakeholders and translating their needs into solutions that integrate in the IT Fabric of the bank. The focus is on the completion of the team's backlog in the order that has been agreed between the various stakeholders and the Product Owner based the added value for the (internal or external) client.
 
Technical experience:

    Strong experience in Django
    experience in software engineering in Python
    GIT (Version Control System).
    Experience in MLOps or DevOps
    experience with testing frameworks (Gherkin, Behave)
    Polyglot (knowledge of other programming languages)
"""
applicant_cv = d.get("experience")

deepseek = Together(
    model="deepseek-ai/DeepSeek-R1",
    together_api_key=os.environ.get('TOGHETER_AI_API_KEY', ''),
    max_tokens=2000,
    temperature=0.2,
    top_p=0.9,
    top_k=40,
    repetition_penalty=1.0,
)

system_prompt_rate_projects = """
You are an expert CV analyst tasked with evaluating the relevance of a job applicant's projects to a specific job 
description. Analyze the Job Description (JD) and each project in the applicant's CV thoroughly. 
Follow these steps:\n\n
1. **Extract Key JD Requirements**: Identify required skills, technologies, methodologies, 
industry experience, responsibilities, and seniority level from the JD.\n
2. **Evaluate Each Project**: For each project, assess:\n
  - Technical skills/tools used (direct or adjacent matches to JD)\n
  - Industry alignment\n   
  - Methodologies/processes (e.g., Agile, CI/CD)\n   
  - Responsibilities/role depth (match with JD's seniority)\n   
  - Problem-solving scope (complexity vs JD requirements)\n
3. **Score Strictly**:\n  
- 100: Perfect match (all key JD elements present)\n   
- 80: Strong relevance (core requirements addressed)\n   
- 60: Partial relevance (some JD elements present)\n   
- 40: Minimal relevance (tangential connections)\n   
- ≤30: No relevance\n4. **Mandatory Inclusivity**: 
Score ALL projects even with 0 relevance.\n
5. **Output**: Valid JSON array with {project-id, score, reason}. 
No markdown.\n\nBe critical—a 'Frontend Developer' project scores 30 for a 'Data Engineer' role unless 
transferable skills exist. 
Prioritize concrete matches over implied ones. Consider project recency only if JD emphasizes it.
"""
prompt_template = """
Job Description: {job_description}
---
Applicant's CV: {applicant_cv}
"""
prompt = prompt_template.format(job_description=job_description, applicant_cv=applicant_cv)
print(prompt)

messages = [
    SystemMessage(content=system_prompt_rate_projects),
    HumanMessage(content=prompt),
]

result = ''
while True:
    print('looping')
    try:
        result = deepseek.invoke(
            messages,
            stream=False,
        )
    except Exception as e:
        if "503" in str(e):
            print('Service Unavailable. Retrying in 15 ...')
            time.sleep(15)
            continue
    break
print(result)
print(replace_think_tags(result))
