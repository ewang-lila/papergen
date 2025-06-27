LLM_PROMPT_TEMPLATE = """
You are an expert researcher in the sciences. 

## Task  
You are provided a published research paper in physics. The paper's main results can be divided into two clear sections. 

Your task: create ONE problem that captures the {segment} of the paper's main mathematical and physical work as thoroughly as possible. (Note that some papers place their main results in the appendix; if this is the case, also consider the results from the appendix in your problem.) There is already a problem for the other half of the paper, and each problem should focus on a different aspect of the results, so make sure your problem captures the area you have been given.

---

## Strict requirements  

- Independence. Every question must be *fully self-contained* and exactly follow the steps provided in the paper: embed all notation, assumptions, and context that the student needs *inside that single question*. Under no circumstances should you refer to a "previous problem", "same system", "as shown in the paper", "the appendix", etc. There should be no references to any material from the paper in the problem statement, since your problem statement should fully capture the work of the paper.
- Focus. Each question must ask the student to derive one analytical result that appears in the paper (e.g. a specific equation). Do not ask to "show," "prove," or "verify" a result, since the grader only considers the student's final result. The question also may not rely on data analysis; it should only involve derivations and reasoning.
- Difficulty. A problem should require many steps and have a unique, objectively checkable answer that is directly written in the paper. I.e., every problem must have a solution that directly corresponds to a result in the paper. The problem must be mathematically nontrivial and as challenging as possible, but still self-contained and solvable. The final answer should be a mathematical expression, not a simple number or single variable. Do **NOT** choose minor results like definitions, substitutions, or auxiliary bounds. The final solution expression should be one of the most complex results in the paper.
- Equation-oriented. The problem should guide students through deriving a specific result that appears in the paper, not proving a general concept.
- Non-duplication. Since you're creating problem {problem_number} of 2, ensure this problem focuses on a different mathematical step than what would be natural for the other problems. If you are writing problem 2, you can include results dependent on the first half of the paper as long as you rewrite *all* the required context for the problem.
- Clarity. Begin each question with a short "Background" paragraph that defines all symbols and states all assumptions used later in that question, as well as some context. End with a "Task" sentence that states exactly what the student must show, without revealing the final expression. Your problem should ask for exactly *one* expression.
- Solutions section. After each question, give the ground-truth solution expression from the paper. Only output the *final solution* in *one* LaTeX \boxed{{}} expression, after the "Solution:" text. There should be no other text in the \boxed{{}} expression other than the final, simplified solution.
- No extraneous parts. Omit numerical verification, coding exercises, open-ended extensions, grading rubrics, etc. Write only the problem (i.e., the background and task) and the solution. DO NOT INCLUDE ANYTHING ELSE IN YOUR RESPONSE!
- Format everything in proper Markdown and LaTeX code. Do not use any special characters in your response; use LaTeX commands for all symbols and characters.


This is the template you must follow to provide the problem statement and solution:

---

Background:  

Task:

Solution:

---

Following these instructions, read the attached paper and create the problem.""" 