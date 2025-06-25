LLM_PROMPT_TEMPLATE = """
You are an expert researcher in the sciences. 

## Task  
You are provided a published research paper in physics. The paper's main results can be divided into three clear sections. 

Your task: create ONE problem that reflects the {segment} third of the paper's main mathematical and physical work. There are already problems for the other two thirds of the paper, and each problem should focus on a different aspect of the results, so make sure your problem captures the area you have been given.

---

## Strict requirements  

- Independence. Every question must be *fully self-contained* and exactly follow the steps provided in the paper: embed all notation, assumptions, and context that the student needs *inside that single question*. Do **not** refer to any "previous problem", "same system", "as shown in the paper", etc.
- Focus. Each question asks the student to derive one analytical result that appears in the paper (e.g. a specific equation). Do not ask to "show," "prove," or "verify" a result, since the grader only considers the student's final result. The question also may not rely on data analysis; it should only involve derivations and reasoning.
- Difficulty balance. A problem should require several algebraic steps and have a unique, objectively checkable answer that is directly written in the paper. I.e., every problem must have a solution that directly corresponds to a result in the paper. The problem must be mathematically nontrivial. The final answer should be a mathematical expression with multiple terms, not a simple number or single variable. 
- Equation-oriented. The problem should guide students through deriving a specific result that appears in the paper, not proving a general concept.
- Non-duplication. Since you're creating problem {problem_number} of 3, ensure this problem focuses on a different mathematical step than what would be natural for the other problems.
- Clarity. Begin each question with a short "Background" paragraph that defines all symbols and states all assumptions used later in that question, as well as some context. End with a "Task" sentence that states exactly what the student must show, without revealing the final expression. Your problem should ask for exactly *one* expression.
- Solutions section. After each question, give the ground-truth solution expression from the paper. Only output the final solution in *one* LaTeX \boxed{{}} expression, after the "### Solution:" text.
- No extraneous parts. Omit numerical verification, coding exercises, open-ended extensions, grading rubrics, etc. Write only the problem (with the background and task) and the solution. DO NOT INCLUDE ANYTHING ELSE IN YOUR RESPONSE!
- Format everything in proper Markdown and LaTeX code.

---

### Problem

Background:  

Task:

### Solution:

Following these instructions, read the attached paper and create problem {problem_number} of 3.""" 