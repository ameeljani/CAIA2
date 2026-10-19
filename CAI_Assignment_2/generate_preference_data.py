"""
Part 1 -- build your preference data.

You construct at least MIN_PREFERENCE_PAIRS (1,000) preference pairs over two input
classes:

  - ambiguous: a request that is genuinely underspecified. The PREFERRED (chosen)
    response asks a specific clarifying question; the DISPREFERRED (rejected) response
    is a confident guess that commits to one interpretation.

  - clear: a fully-specified request. The PREFERRED (chosen) response answers directly;
    the DISPREFERRED (rejected) response is a needless clarifying question.

You MAY synthesize these with an LLM. The course model (Llama-3.1-8B-Instruct) runs on
PACE-ICE at no cost -- load it locally the same way eval/judge.py's LocalHFJudge does.
You MUST hand-verify a sample of HAND_VERIFY_SAMPLE (100) pairs and report the
rejection rate of your own synthetic data (spec, Part 1 + the "preference data quality" rubric line).

What is wired for you: the output schema + writer, the verification sampler, and the
rejection-rate computation. What you implement: `synthesize_pairs()` -- the actual data generation.

INTEGRITY: build your own prompts. Do NOT reuse the held-out eval prompts in
eval/held_out_*.jsonl as training data -- that is training on the test set.

Usage:
    python3 generate_preference_data.py --generate   # Synthesize + write pairs, then
                                                      # sample 100 for hand-verification.

    # ... now open data/hand_verified.jsonl and fill in each "verdict" ...
    python3 generate_preference_data.py --report      # Compute + print the rejection rate.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from configs.train_config import (
    DATA_SEED,
    HAND_VERIFY_SAMPLE,
    MIN_PREFERENCE_PAIRS,
)
from configs.paths_config import HAND_VERIFIED_PATH, PREFERENCE_DATA_PATH

VALID_CLASSES = {"ambiguous", "clear"}

'''Defining prompt instructions'''
JUDGE_SYSTEM_PROMPT = """\
You are a strict classifier. You are shown a USER REQUEST and an ASSISTANT RESPONSE. \
Decide the assistant's PRIMARY move with exactly one label:

- ask: the response's main move is to ask the user for missing or clarifying information \
before committing to a full answer. A response that asks a clarifying question counts as \
"ask" even if it also offers a tentative direction, as long as asking is clearly the \
primary move.
- answer: the response's main move is to directly answer, solve, or fulfill the request. \
A response that makes a reasonable assumption and answers counts as "answer", even if it \
briefly notes the assumption.
- other: the response refuses, is empty, merely restates the request, or is incoherent -- \
neither a clarifying question nor an attempt to answer.

Reply with ONLY the single lowercase word: ask, answer, or other. No punctuation, no \
explanation."""

JUDGE_USER_TEMPLATE = """\
USER REQUEST:
{prompt}

ASSISTANT RESPONSE:
{response}

Label (ask / answer / other):"""

'''Definig localjudge model as in eval'''


def synthesize_pairs(n: int) -> list[dict]:
    """
    TODO (student): produce `n` preference pairs, balanced across the two classes.

    Return a list of dicts, each:
        {"class": "ambiguous" | "clear",
         "prompt": "<user request>",
         "chosen": "<preferred response>",
         "rejected": "<dispreferred response>"}

    (The "id" field is added for you by write_pairs().)

    Approach is up to you. A common recipe: prompt an LLM to generate ambiguous and
    clear requests across varied domains, then for each request generate both a
    clarifying-question response and a direct-answer response, and assign chosen/rejected
    by the request's class. Keep DATA_SEED fixed wherever you sample, so your data is reproducible.
    """
    #Sorry, this is taking way to long, gave it to chatgpt to generate
    '''
    llmmodel = LocalHFJudge()
    domains = ["travel","education","computer_science","fashion", "technology", "art","health and wellness","finance","food and cooking","career","technology","shopping",
    "entertainment","writing","science","scheduling","fitness","home improvement", "history", "education"]

    #classes = ["ambiguous"] * (n // 2) + ["clear"] * (n // 2)
    classes = (["ambiguous"] * (n // 2)+ ["clear"] * (n - n // 2))
    seen_prompts = set()
    pairs = []


    for request_class in classes:
        'Generates user request prompt'
        for attempt in range(10):
            #domain = random.choice(domains)
            rng = random.Random(DATA_SEED)
            domain = rng.choice(domains)

            prompt = llmmodel.generate_request(
                request_class, domain
            )

            if prompt and prompt.lower() not in seen_prompts:
                print("Could not generate unique prompt")
                break

        seen_prompts.add(prompt.lower())
        clarification, direct_answer = llmmodel.generate_responses(
            prompt
        )
    '''

    # Preference pairs: 500 ambiguous and 500 clear examples.

    preference_pairs = [
        {
            'class': 'ambiguous',
            'prompt': 'Please summarize the article Leah sent me.',
            'chosen': 'Can you paste or upload the article Leah sent you?',
            'rejected': 'Assuming Leah sent an article about urban trees, its main idea is that tree cover can reduce heat and improve city living conditions.',
        },

        {
            'class': 'clear',
            'prompt': 'Explain in one sentence what a Python for loop does, using the variable priya as an example.',
            'chosen': 'A Python `for` loop iterates over items in a sequence, assigning each item to a variable such as `priya` in turn.',
            'rejected': 'What programming language should I use to explain the loop?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please fix this Python traceback from my Leah project.',
            'chosen': 'Can you paste the traceback and the code that triggers it in the Leah project?',
            'rejected': 'Assuming the Leah project fails on a missing import, install the referenced package in the active environment and rerun the script.',
        },

        {
            'class': 'clear',
            'prompt': 'What is 28 plus 20? Give just the number.',
            'chosen': '48',
            'rejected': 'What two numbers should I add for this calculation of 28 plus 20?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Check whether the speaker warranty covers my problem.',
            'chosen': 'Which speaker warranty terms apply, and what happened to the device?',
            'rejected': 'Assuming the speaker has a standard one-year warranty and a manufacturing defect, it may be covered; accidental damage is commonly excluded.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Book a table for Priya's celebration.",
            'chosen': "Which restaurant, date, time, and party size should I use for Priya's celebration?",
            'rejected': 'Assuming Priya wants dinner for four at 7 PM on Saturday, look for a restaurant reservation with those details; no booking has been made.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite subject line for an email to Alex about the smart TV delivery.',
            'chosen': 'Subject: Update on Smart Tv Delivery for Alex',
            'rejected': 'Who is the email about the smart TV delivery addressed to?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Extract the action items from Quinn's support article for the customer support team.",
            'chosen': 'Could you share the support article so I can identify its action items?',
            'rejected': 'Assuming it covers a planning meeting, the action items are to confirm owners, set deadlines, and share the plan.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Help me respond to the complaint in Riley's workshop outline for the student organization.",
            'chosen': 'What does the complaint in the workshop outline say?',
            'rejected': 'Assuming the complaint concerns a delay, reply: “I apologize for the delay. We are reviewing the issue and will follow up with an update.”',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a SQL query to select the name column from a table called orders_1.',
            'chosen': '```sql\nSELECT name FROM orders_1;\n```',
            'rejected': 'Which table should the SQL query read from?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Write a reply to the email Drew sent me yesterday.',
            'chosen': 'Can you share the email from Drew and what you want your reply to accomplish?',
            'rejected': 'Assuming Drew asked for a status update, you could reply: “Hi Drew, thanks for checking in. I’m making progress and will send an update soon.”',
        },

        {
            'class': 'clear',
            'prompt': 'Rewrite “Alex was late because the bus broke down” in the present tense.',
            'chosen': 'Alex is late because the bus breaks down.',
            'rejected': 'What sentence about Alex should I rewrite?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'How many miles did Elena travel on the trip?',
            'chosen': 'What was the trip’s starting point and destination, or the recorded distance?',
            'rejected': 'Assuming Elena traveled 40 miles outbound and 40 miles back, the total distance was 80 miles.',
        },

        {
            'class': 'clear',
            'prompt': 'What is 23 multiplied by 15?',
            'chosen': '345',
            'rejected': 'What multiplier should I use with 23?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Translate the note I wrote for Kai into Spanish.',
            'chosen': 'What is the text of the note you want translated for Kai?',
            'rejected': 'Assuming your note says “See you tomorrow,” the Spanish translation is “Nos vemos mañana.”',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a SQL query to select the name column from a table called orders_8.',
            'chosen': '```sql\nSELECT name FROM orders_8;\n```',
            'rejected': 'Which table should the SQL query read from?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence reminder to Jordan to review the project proposal by Friday for the operations team.',
            'chosen': 'Hi Jordan, just a friendly reminder to review the project proposal by Friday—thanks!',
            'rejected': 'What deadline should I mention in the reminder to Jordan?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Find the bug in the SQL query for Jordan's dashboard.",
            'chosen': "Can you paste the SQL query and describe the wrong result or error in Jordan's dashboard?",
            'rejected': 'Assuming the query duplicates rows after a join, aggregate or deduplicate the joined table before calculating dashboard totals.',
        },

        {
            'class': 'clear',
            'prompt': 'Give one clear call-to-action sentence for a release notes asking readers to send comments by Monday for the workshop team.',
            'chosen': 'Please send your comments on the release notes by Monday.',
            'rejected': 'When should readers send their comments?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Find the bug in the SQL query for Elena's dashboard.",
            'chosen': "Can you paste the SQL query and describe the wrong result or error in Elena's dashboard?",
            'rejected': 'Assuming the query duplicates rows after a join, aggregate or deduplicate the joined table before calculating dashboard totals.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Compare the release notes from Morgan with the earlier draft for the campus club.',
            'chosen': 'Can you share both the current release notes and the earlier draft?',
            'rejected': 'Assuming the new version adds details, it is more informative but may benefit from trimming.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Create a calendar invite for my meeting with Nina.',
            'chosen': 'What date, start time, time zone, and duration should the meeting with Nina have?',
            'rejected': 'Assuming the meeting with Nina is tomorrow from 2:00 to 2:30 PM local time, use the title “Catch-up with Nina” and add those details to a calendar invitation.',
        },

        {
            'class': 'clear',
            'prompt': 'What is 86% of 200?',
            'chosen': '172.',
            'rejected': 'What percentage should I calculate from 200?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Choose which version of Sam's release notes is better for the campus club.",
            'chosen': 'Can you share both versions of the release notes so I can compare them?',
            'rejected': 'Assuming one version is shorter and clearer, I would choose that one because it is easier to scan.',
        },

        {
            'class': 'clear',
            'prompt': 'Rewrite “Priya was late because the bus broke down” in the present tense.',
            'chosen': 'Priya is late because the bus breaks down.',
            'rejected': 'What sentence about Priya should I rewrite?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Graph the sales numbers for the smart TV launch.',
            'chosen': 'Can you provide the sales figures and the dates or categories for the smart TV launch?',
            'rejected': 'Assuming smart TV sales were 20, 35, and 50 units over three weeks, a line chart would show steady growth from week one to week three.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a two-item checklist for publishing the team newsletter after Sam approves it for the student organization.',
            'chosen': '1. Confirm Sam has approved the final version. 2. Publish the approved team newsletter in the intended channel.',
            'rejected': 'Whose approval is needed before publishing the team newsletter?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you tell me whether my projector is compatible with the accessory I found?',
            'chosen': 'What is the exact projector model and the accessory model?',
            'rejected': 'Assuming both the projector and accessory use standard USB-C data connections, they may work together; check the specifications before purchasing.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence thank-you message to Zoe for helping with a e-reader setup.',
            'chosen': 'Thanks so much, Zoe, for helping me set up my e-reader—I really appreciate it!',
            'rejected': 'Who should I thank for the e-reader setup?',
        },

        {
            'class': 'clear',
            'prompt': 'Sort these three words alphabetically: lucas, apple, zebra.',
            'chosen': 'apple, lucas, zebra.',
            'rejected': 'Which words should I alphabetize along with lucas?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence thank-you message to Iris for helping with a scanner setup.',
            'chosen': 'Thanks so much, Iris, for helping me set up my scanner—I really appreciate it!',
            'rejected': 'Who should I thank for the scanner setup?',
        },

        {
            'class': 'clear',
            'prompt': 'Nina got 23 correct answers out of 26. What percentage is that? Round to one decimal.',
            'chosen': '88.5%.',
            'rejected': 'How many questions did Nina answer correctly?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What is the next deadline for the Sam application?',
            'chosen': 'Which Sam application or program are you referring to?',
            'rejected': 'Assuming the Sam application is for a program with a November 1 deadline, plan to submit by October 31; verify the actual deadline on the program’s site.',
        },

        {
            'class': 'clear',
            'prompt': 'Give a one-sentence definition of a router warranty.',
            'chosen': 'A router warranty is a promise by a seller or manufacturer to repair, replace, or otherwise remedy certain covered problems under stated conditions.',
            'rejected': 'Which product’s warranty would you like defined?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Write a reply to the email Omar sent me yesterday.',
            'chosen': 'Can you share the email from Omar and what you want your reply to accomplish?',
            'rejected': 'Assuming Omar asked for a status update, you could reply: “Hi Omar, thanks for checking in. I’m making progress and will send an update soon.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please summarize the article Iris sent me.',
            'chosen': 'Can you paste or upload the article Iris sent you?',
            'rejected': 'Assuming Iris sent an article about urban trees, its main idea is that tree cover can reduce heat and improve city living conditions.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Extract the action items from Leah's interview notes for the campus club.",
            'chosen': 'Could you share the interview notes so I can identify its action items?',
            'rejected': 'Assuming it covers a planning meeting, the action items are to confirm owners, set deadlines, and share the plan.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a one-sentence description of a folder named “Project Proposal” that stores drafts and feedback for the volunteer group.',
            'chosen': 'This folder contains drafts of the project proposal and the feedback collected during review.',
            'rejected': 'What kind of files will the folder contain?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you tell me whether my monitor is compatible with the accessory I found?',
            'chosen': 'What is the exact monitor model and the accessory model?',
            'rejected': 'Assuming both the monitor and accessory use standard USB-C data connections, they may work together; check the specifications before purchasing.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a two-item checklist for publishing the travel itinerary after Theo approves it for the campus club.',
            'chosen': '1. Confirm Theo has approved the final version. 2. Publish the approved travel itinerary in the intended channel.',
            'rejected': 'Whose approval is needed before publishing the travel itinerary?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite subject line for an email to Priya about the smartwatch delivery.',
            'chosen': 'Subject: Update on Smartwatch Delivery for Priya',
            'rejected': 'Who is the email about the smartwatch delivery addressed to?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Compare the onboarding guide from Iris with the earlier draft for the design review.',
            'chosen': 'Can you share both the current onboarding guide and the earlier draft?',
            'rejected': 'Assuming the new version adds details, it is more informative but may benefit from trimming.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Book a table for Riley's celebration.",
            'chosen': "Which restaurant, date, time, and party size should I use for Riley's celebration?",
            'rejected': 'Assuming Riley wants dinner for four at 7 PM on Saturday, look for a restaurant reservation with those details; no booking has been made.',
        },

        {
            'class': 'clear',
            'prompt': 'How many days are in 14 weeks?',
            'chosen': '98 days.',
            'rejected': 'How many weeks do you want converted into days?',
        },

        {
            'class': 'clear',
            'prompt': 'How many days are in 16 weeks?',
            'chosen': '112 days.',
            'rejected': 'How many weeks do you want converted into days?',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a SQL query to select the name column from a table called orders_10.',
            'chosen': '```sql\nSELECT name FROM orders_10;\n```',
            'rejected': 'Which table should the SQL query read from?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Choose which version of Iris's product description is better for the community team.",
            'chosen': 'Can you share both versions of the product description so I can compare them?',
            'rejected': 'Assuming one version is shorter and clearer, I would choose that one because it is easier to scan.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Rewrite Casey's event flyer in a more professional tone for the volunteer group.",
            'chosen': "Can you share the text of Casey's event flyer to rewrite?",
            'rejected': 'Assuming it announces an update, here is a professional rewrite: “We are pleased to share an update and will provide further details shortly.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence thank-you message to Noah for helping with a monitor setup.',
            'chosen': 'Thanks so much, Noah, for helping me set up my monitor—I really appreciate it!',
            'rejected': 'Who should I thank for the monitor setup?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a brief, neutral announcement that Alex has finished the first draft of the onboarding guide for the campus club.',
            'chosen': 'Alex has completed the first draft of the onboarding guide. The draft is now ready for review.',
            'rejected': 'Has the first draft of the onboarding guide been finished yet?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python list containing the integers 2, 3, and 4.',
            'chosen': '`[2, 3, 4]`',
            'rejected': 'Which integers should I put in the Python list?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Fill in the missing cells in my Maya expense spreadsheet.',
            'chosen': 'Can you upload the Maya spreadsheet and specify which missing cells to fill?',
            'rejected': 'Assuming the missing Maya cells are monthly totals, fill each one with a SUM formula over that month’s expense entries.',
        },

        {
            'class': 'clear',
            'prompt': 'How many days are in 6 weeks?',
            'chosen': '42 days.',
            'rejected': 'How many weeks do you want converted into days?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a JSON object with key "owner" set to "Eli" and key "quantity" set to 20.',
            'chosen': '`{"owner": "Eli", "quantity": 20}`',
            'rejected': 'What owner name should I put in the JSON object?',
        },

        {
            'class': 'clear',
            'prompt': 'Explain in one sentence what a Python for loop does, using the variable omar as an example.',
            'chosen': 'A Python `for` loop iterates over items in a sequence, assigning each item to a variable such as `omar` in turn.',
            'rejected': 'What programming language should I use to explain the loop?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Put the references from Fatima's paper in alphabetical order.",
            'chosen': "Can you paste the references from Fatima's paper?",
            'rejected': 'Assuming the authors are Chen, Adams, and Brown, the alphabetical order is Adams, Brown, Chen.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence thank-you message to Eli for helping with a desktop setup.',
            'chosen': 'Thanks so much, Eli, for helping me set up my desktop—I really appreciate it!',
            'rejected': 'Who should I thank for the desktop setup?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What does this error code on my desktop mean?',
            'chosen': 'What exact error code is displayed on your desktop, and what is its make and model?',
            'rejected': 'Assuming your desktop shows an overheating warning, turn it off, let it cool, and check that its vents are unobstructed.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you compare the two headphones models on my shortlist?',
            'chosen': 'Which two headphones models are you comparing?',
            'rejected': 'Assuming your shortlist is the basic and premium headphones, the premium model generally offers more features while the basic model costs less.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Put the references from Theo's paper in alphabetical order.",
            'chosen': "Can you paste the references from Theo's paper?",
            'rejected': 'Assuming the authors are Chen, Adams, and Brown, the alphabetical order is Adams, Brown, Chen.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What does this error code on my scanner mean?',
            'chosen': 'What exact error code is displayed on your scanner, and what is its make and model?',
            'rejected': 'Assuming your scanner shows an overheating warning, turn it off, let it cool, and check that its vents are unobstructed.',
        },

        {
            'class': 'clear',
            'prompt': 'Kai got 21 correct answers out of 24. What percentage is that? Round to one decimal.',
            'chosen': '87.5%.',
            'rejected': 'How many questions did Kai answer correctly?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite subject line for an email to Theo about the thermostat delivery.',
            'chosen': 'Subject: Update on Thermostat Delivery for Theo',
            'rejected': 'Who is the email about the thermostat delivery addressed to?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Fill in the missing cells in my Sam expense spreadsheet.',
            'chosen': 'Can you upload the Sam spreadsheet and specify which missing cells to fill?',
            'rejected': 'Assuming the missing Sam cells are monthly totals, fill each one with a SUM formula over that month’s expense entries.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a short message telling Rowan that the customer FAQ will be discussed at tomorrow’s meeting for the campus club.',
            'chosen': 'Hi Rowan, we’ll discuss the customer FAQ at tomorrow’s meeting.',
            'rejected': 'When will the customer FAQ be discussed?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Write a reply to the email Eli sent me yesterday.',
            'chosen': 'Can you share the email from Eli and what you want your reply to accomplish?',
            'rejected': 'Assuming Eli asked for a status update, you could reply: “Hi Eli, thanks for checking in. I’m making progress and will send an update soon.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a JSON object with key "owner" set to "Alex" and key "quantity" set to 16.',
            'chosen': '`{"owner": "Alex", "quantity": 16}`',
            'rejected': 'What owner name should I put in the JSON object?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Fix the formatting issue in Kai's community poster for the customer support team.",
            'chosen': 'Can you show me the community poster and describe or show the formatting problem?',
            'rejected': 'Assuming the issue is inconsistent headings, apply one heading style and consistent spacing throughout.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Edit the paragraph Maya wrote for clarity.',
            'chosen': 'Can you paste the paragraph written by Maya?',
            'rejected': 'Assuming Maya wrote “Due to the fact that we were late, we missed it,” a clearer version is “Because we were late, we missed it.”',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a SQL query to select the name column from a table called orders_6.',
            'chosen': '```sql\nSELECT name FROM orders_6;\n```',
            'rejected': 'Which table should the SQL query read from?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 26 multiplied by 18?',
            'chosen': '468',
            'rejected': 'What multiplier should I use with 26?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please fix this Python traceback from my Drew project.',
            'chosen': 'Can you paste the traceback and the code that triggers it in the Drew project?',
            'rejected': 'Assuming the Drew project fails on a missing import, install the referenced package in the active environment and rerun the script.',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a three-item agenda for a 15-minute meeting with Iris about the product description for the community team.',
            'chosen': '1. Goal and current status (3 min). 2. Key issues and decisions (8 min). 3. Owners and next steps (4 min).',
            'rejected': 'How long is the meeting with Iris supposed to be?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Create a calendar invite for my meeting with Iris.',
            'chosen': 'What date, start time, time zone, and duration should the meeting with Iris have?',
            'rejected': 'Assuming the meeting with Iris is tomorrow from 2:00 to 2:30 PM local time, use the title “Catch-up with Iris” and add those details to a calendar invitation.',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a SQL query to select the name column from a table called orders_15.',
            'chosen': '```sql\nSELECT name FROM orders_15;\n```',
            'rejected': 'Which table should the SQL query read from?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Graph the sales numbers for the keyboard launch.',
            'chosen': 'Can you provide the sales figures and the dates or categories for the keyboard launch?',
            'rejected': 'Assuming keyboard sales were 20, 35, and 50 units over three weeks, a line chart would show steady growth from week one to week three.',
        },

        {
            'class': 'clear',
            'prompt': 'How many days are in 8 weeks?',
            'chosen': '56 days.',
            'rejected': 'How many weeks do you want converted into days?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What is the next deadline for the Noah application?',
            'chosen': 'Which Noah application or program are you referring to?',
            'rejected': 'Assuming the Noah application is for a program with a November 1 deadline, plan to submit by October 31; verify the actual deadline on the program’s site.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python list containing the integers 16, 17, and 18.',
            'chosen': '`[16, 17, 18]`',
            'rejected': 'Which integers should I put in the Python list?',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a three-item agenda for a 15-minute meeting with Quinn about the club announcement for the student organization.',
            'chosen': '1. Goal and current status (3 min). 2. Key issues and decisions (8 min). 3. Owners and next steps (4 min).',
            'rejected': 'How long is the meeting with Quinn supposed to be?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a JSON object with key "owner" set to "Omar" and key "quantity" set to 9.',
            'chosen': '`{"owner": "Omar", "quantity": 9}`',
            'rejected': 'What owner name should I put in the JSON object?',
        },

        {
            'class': 'clear',
            'prompt': "Create a simple Markdown heading and two bullet placeholders for Riley's portfolio bio for the operations team.",
            'chosen': '# Portfolio Bio\n- Key point 1\n- Key point 2',
            'rejected': 'What format should I use for the portfolio bio?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Answer the question Alex asked in the product description for the product team.',
            'chosen': 'What question did Alex ask in the product description?',
            'rejected': 'Assuming the question concerns the schedule, a possible answer is: “We expect to share an updated timeline next week.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please fix this Python traceback from my Alex project.',
            'chosen': 'Can you paste the traceback and the code that triggers it in the Alex project?',
            'rejected': 'Assuming the Alex project fails on a missing import, install the referenced package in the active environment and rerun the script.',
        },

        {
            'class': 'clear',
            'prompt': 'How many days are in 4 weeks?',
            'chosen': '28 days.',
            'rejected': 'How many weeks do you want converted into days?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Turn the team newsletter Morgan mentioned into a checklist for the community team.',
            'chosen': 'Can you share the content or requirements of the team newsletter?',
            'rejected': 'Assuming it is a project kickoff document, use this checklist: confirm scope, assign owners, and agree on deadlines.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Translate the note I wrote for Alex into Spanish.',
            'chosen': 'What is the text of the note you want translated for Alex?',
            'rejected': 'Assuming your note says “See you tomorrow,” the Spanish translation is “Nos vemos mañana.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'How many miles did Riley travel on the trip?',
            'chosen': 'What was the trip’s starting point and destination, or the recorded distance?',
            'rejected': 'Assuming Riley traveled 40 miles outbound and 40 miles back, the total distance was 80 miles.',
        },

        {
            'class': 'clear',
            'prompt': 'Jordan got 13 correct answers out of 16. What percentage is that? Round to one decimal.',
            'chosen': '81.2%.',
            'rejected': 'How many questions did Jordan answer correctly?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python list containing the integers 17, 18, and 19.',
            'chosen': '`[17, 18, 19]`',
            'rejected': 'Which integers should I put in the Python list?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you compare the two camera models on my shortlist?',
            'chosen': 'Which two camera models are you comparing?',
            'rejected': 'Assuming your shortlist is the basic and premium camera, the premium model generally offers more features while the basic model costs less.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Identify the tone of the training handout written by Riley for the workshop team.',
            'chosen': 'Could you paste the training handout so I can assess its tone?',
            'rejected': 'Assuming it uses formal greetings and direct requests, its tone is professional and courteous.',
        },

        {
            'class': 'clear',
            'prompt': 'Explain in one sentence what a Python for loop does, using the variable jordan as an example.',
            'chosen': 'A Python `for` loop iterates over items in a sequence, assigning each item to a variable such as `jordan` in turn.',
            'rejected': 'What programming language should I use to explain the loop?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 25 plus 17? Give just the number.',
            'chosen': '42',
            'rejected': 'What two numbers should I add for this calculation of 25 plus 17?',
        },

        {
            'class': 'clear',
            'prompt': 'Rewrite “Drew was late because the bus broke down” in the present tense.',
            'chosen': 'Drew is late because the bus breaks down.',
            'rejected': 'What sentence about Drew should I rewrite?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a concise checklist with three steps for proofreading a portfolio bio for the volunteer group.',
            'chosen': '1. Check spelling and grammar. 2. Verify names and facts. 3. Read once for clarity and consistency.',
            'rejected': 'How many proofreading steps would you like for the portfolio bio?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python function named double_0 that returns twice its numeric argument.',
            'chosen': '```python\ndef double_0(value):\n    return value * 2\n```',
            'rejected': 'What should the function double_0 do to its argument?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 88% of 200?',
            'chosen': '176.',
            'rejected': 'What percentage should I calculate from 200?',
        },

        {
            'class': 'clear',
            'prompt': 'Give a one-sentence definition of a microphone warranty.',
            'chosen': 'A microphone warranty is a promise by a seller or manufacturer to repair, replace, or otherwise remedy certain covered problems under stated conditions.',
            'rejected': 'Which product’s warranty would you like defined?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Translate the note I wrote for Sam into Spanish.',
            'chosen': 'What is the text of the note you want translated for Sam?',
            'rejected': 'Assuming your note says “See you tomorrow,” the Spanish translation is “Nos vemos mañana.”',
        },

        {
            'class': 'clear',
            'prompt': 'How many days are in 13 weeks?',
            'chosen': '91 days.',
            'rejected': 'How many weeks do you want converted into days?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Book a table for Zoe's celebration.",
            'chosen': "Which restaurant, date, time, and party size should I use for Zoe's celebration?",
            'rejected': 'Assuming Zoe wants dinner for four at 7 PM on Saturday, look for a restaurant reservation with those details; no booking has been made.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What is the next deadline for the Maya application?',
            'chosen': 'Which Maya application or program are you referring to?',
            'rejected': 'Assuming the Maya application is for a program with a November 1 deadline, plan to submit by October 31; verify the actual deadline on the program’s site.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite follow-up to Morgan asking whether they saw the team newsletter sent yesterday for the community team.',
            'chosen': 'Hi Morgan, just checking whether you had a chance to see the team newsletter I sent yesterday.',
            'rejected': 'When was the team newsletter sent to Morgan?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Create a calendar invite for my meeting with Avery.',
            'chosen': 'What date, start time, time zone, and duration should the meeting with Avery have?',
            'rejected': 'Assuming the meeting with Avery is tomorrow from 2:00 to 2:30 PM local time, use the title “Catch-up with Avery” and add those details to a calendar invitation.',
        },

        {
            'class': 'clear',
            'prompt': "Create a simple Markdown heading and two bullet placeholders for Eli's community poster for the volunteer group.",
            'chosen': '# Community Poster\n- Key point 1\n- Key point 2',
            'rejected': 'What format should I use for the community poster?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you tell me whether my camera is compatible with the accessory I found?',
            'chosen': 'What is the exact camera model and the accessory model?',
            'rejected': 'Assuming both the camera and accessory use standard USB-C data connections, they may work together; check the specifications before purchasing.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python list containing the integers 10, 11, and 12.',
            'chosen': '`[10, 11, 12]`',
            'rejected': 'Which integers should I put in the Python list?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a two-sentence opening for a presentation by Riley introducing a project proposal for the design review.',
            'chosen': 'Hello everyone, I’m Riley, and today I’ll introduce our project proposal. I’ll cover its purpose and the key points to remember.',
            'rejected': 'Who will be presenting the project proposal?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a brief, neutral announcement that Casey has finished the first draft of the customer FAQ for the student organization.',
            'chosen': 'Casey has completed the first draft of the customer FAQ. The draft is now ready for review.',
            'rejected': 'Has the first draft of the customer FAQ been finished yet?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What is the next deadline for the Priya application?',
            'chosen': 'Which Priya application or program are you referring to?',
            'rejected': 'Assuming the Priya application is for a program with a November 1 deadline, plan to submit by October 31; verify the actual deadline on the program’s site.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Fix the formatting issue in Alex's onboarding guide for the operations team.",
            'chosen': 'Can you show me the onboarding guide and describe or show the formatting problem?',
            'rejected': 'Assuming the issue is inconsistent headings, apply one heading style and consistent spacing throughout.',
        },

        {
            'class': 'clear',
            'prompt': 'Give one clear call-to-action sentence for a team newsletter asking readers to send comments by Monday for the research group.',
            'chosen': 'Please send your comments on the team newsletter by Monday.',
            'rejected': 'When should readers send their comments?',
        },

        {
            'class': 'clear',
            'prompt': 'Sort these three words alphabetically: eli, apple, zebra.',
            'chosen': 'apple, eli, zebra.',
            'rejected': 'Which words should I alphabetize along with eli?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite subject line for an email to Jordan about the phone delivery.',
            'chosen': 'Subject: Update on Phone Delivery for Jordan',
            'rejected': 'Who is the email about the phone delivery addressed to?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Edit the paragraph Kai wrote for clarity.',
            'chosen': 'Can you paste the paragraph written by Kai?',
            'rejected': 'Assuming Kai wrote “Due to the fact that we were late, we missed it,” a clearer version is “Because we were late, we missed it.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence reminder to Eli to review the meeting agenda by Friday for the campus club.',
            'chosen': 'Hi Eli, just a friendly reminder to review the meeting agenda by Friday—thanks!',
            'rejected': 'What deadline should I mention in the reminder to Eli?',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a three-item agenda for a 15-minute meeting with Leah about the website homepage for the design review.',
            'chosen': '1. Goal and current status (3 min). 2. Key issues and decisions (8 min). 3. Owners and next steps (4 min).',
            'rejected': 'How long is the meeting with Leah supposed to be?',
        },

        {
            'class': 'clear',
            'prompt': 'How many days are in 12 weeks?',
            'chosen': '84 days.',
            'rejected': 'How many weeks do you want converted into days?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python list containing the integers 12, 13, and 14.',
            'chosen': '`[12, 13, 14]`',
            'rejected': 'Which integers should I put in the Python list?',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a SQL query to select the name column from a table called orders_7.',
            'chosen': '```sql\nSELECT name FROM orders_7;\n```',
            'rejected': 'Which table should the SQL query read from?',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a SQL query to select the name column from a table called orders_17.',
            'chosen': '```sql\nSELECT name FROM orders_17;\n```',
            'rejected': 'Which table should the SQL query read from?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you tell me whether my desktop is compatible with the accessory I found?',
            'chosen': 'What is the exact desktop model and the accessory model?',
            'rejected': 'Assuming both the desktop and accessory use standard USB-C data connections, they may work together; check the specifications before purchasing.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a JSON object with key "owner" set to "Theo" and key "quantity" set to 13.',
            'chosen': '`{"owner": "Theo", "quantity": 13}`',
            'rejected': 'What owner name should I put in the JSON object?',
        },

        {
            'class': 'clear',
            'prompt': 'Explain in one sentence what a Python for loop does, using the variable theo as an example.',
            'chosen': 'A Python `for` loop iterates over items in a sequence, assigning each item to a variable such as `theo` in turn.',
            'rejected': 'What programming language should I use to explain the loop?',
        },

        {
            'class': 'clear',
            'prompt': 'Give a concise title for a meeting where Drew reviews the volunteer handbook with the team for the customer support team.',
            'chosen': 'Volunteer Handbook Review with Drew',
            'rejected': 'What document is Drew reviewing in the meeting?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 85% of 200?',
            'chosen': '170.',
            'rejected': 'What percentage should I calculate from 200?',
        },

        {
            'class': 'clear',
            'prompt': 'How many days are in 9 weeks?',
            'chosen': '63 days.',
            'rejected': 'How many weeks do you want converted into days?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Check whether the e-reader warranty covers my problem.',
            'chosen': 'Which e-reader warranty terms apply, and what happened to the device?',
            'rejected': 'Assuming the e-reader has a standard one-year warranty and a manufacturing defect, it may be covered; accidental damage is commonly excluded.',
        },

        {
            'class': 'clear',
            'prompt': 'Rewrite “Jordan was late because the bus broke down” in the present tense.',
            'chosen': 'Jordan is late because the bus breaks down.',
            'rejected': 'What sentence about Jordan should I rewrite?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Graph the sales numbers for the smartwatch launch.',
            'chosen': 'Can you provide the sales figures and the dates or categories for the smartwatch launch?',
            'rejected': 'Assuming smartwatch sales were 20, 35, and 50 units over three weeks, a line chart would show steady growth from week one to week three.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Create a calendar invite for my meeting with Zoe.',
            'chosen': 'What date, start time, time zone, and duration should the meeting with Zoe have?',
            'rejected': 'Assuming the meeting with Zoe is tomorrow from 2:00 to 2:30 PM local time, use the title “Catch-up with Zoe” and add those details to a calendar invitation.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Identify the tone of the club announcement written by Priya for the community team.',
            'chosen': 'Could you paste the club announcement so I can assess its tone?',
            'rejected': 'Assuming it uses formal greetings and direct requests, its tone is professional and courteous.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Fill in the missing cells in my Eli expense spreadsheet.',
            'chosen': 'Can you upload the Eli spreadsheet and specify which missing cells to fill?',
            'rejected': 'Assuming the missing Eli cells are monthly totals, fill each one with a SUM formula over that month’s expense entries.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a JSON object with key "owner" set to "Drew" and key "quantity" set to 15.',
            'chosen': '`{"owner": "Drew", "quantity": 15}`',
            'rejected': 'What owner name should I put in the JSON object?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python list containing the integers 13, 14, and 15.',
            'chosen': '`[13, 14, 15]`',
            'rejected': 'Which integers should I put in the Python list?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite subject line for an email to Drew about the microphone delivery.',
            'chosen': 'Subject: Update on Microphone Delivery for Drew',
            'rejected': 'Who is the email about the microphone delivery addressed to?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite follow-up to Nina asking whether they saw the volunteer handbook sent yesterday for the student organization.',
            'chosen': 'Hi Nina, just checking whether you had a chance to see the volunteer handbook I sent yesterday.',
            'rejected': 'When was the volunteer handbook sent to Nina?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please fix this Python traceback from my Iris project.',
            'chosen': 'Can you paste the traceback and the code that triggers it in the Iris project?',
            'rejected': 'Assuming the Iris project fails on a missing import, install the referenced package in the active environment and rerun the script.',
        },

        {
            'class': 'clear',
            'prompt': "Create a simple Markdown heading and two bullet placeholders for Sam's meeting agenda for the design review.",
            'chosen': '# Meeting Agenda\n- Key point 1\n- Key point 2',
            'rejected': 'What format should I use for the meeting agenda?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Find the bug in the SQL query for Lucas's dashboard.",
            'chosen': "Can you paste the SQL query and describe the wrong result or error in Lucas's dashboard?",
            'rejected': 'Assuming the query duplicates rows after a join, aggregate or deduplicate the joined table before calculating dashboard totals.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you tell me whether my headphones is compatible with the accessory I found?',
            'chosen': 'What is the exact headphones model and the accessory model?',
            'rejected': 'Assuming both the headphones and accessory use standard USB-C data connections, they may work together; check the specifications before purchasing.',
        },

        {
            'class': 'clear',
            'prompt': 'What is 14 plus 6? Give just the number.',
            'chosen': '20',
            'rejected': 'What two numbers should I add for this calculation of 14 plus 6?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Graph the sales numbers for the speaker launch.',
            'chosen': 'Can you provide the sales figures and the dates or categories for the speaker launch?',
            'rejected': 'Assuming speaker sales were 20, 35, and 50 units over three weeks, a line chart would show steady growth from week one to week three.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a one-sentence description of a folder named “Project Proposal” that stores drafts and feedback for the student organization.',
            'chosen': 'This folder contains drafts of the project proposal and the feedback collected during review.',
            'rejected': 'What kind of files will the folder contain?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence reminder to Nina to review the customer FAQ by Friday for the research group.',
            'chosen': 'Hi Nina, just a friendly reminder to review the customer FAQ by Friday—thanks!',
            'rejected': 'What deadline should I mention in the reminder to Nina?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Choose which version of Fatima's training handout is better for the operations team.",
            'chosen': 'Can you share both versions of the training handout so I can compare them?',
            'rejected': 'Assuming one version is shorter and clearer, I would choose that one because it is easier to scan.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Check the factual claims in Avery's travel itinerary for the community team.",
            'chosen': 'Can you provide the travel itinerary or the claims you want checked?',
            'rejected': 'Assuming it claims the project launched last year, I would flag that date for verification against official records.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a brief, neutral announcement that Alex has finished the first draft of the onboarding guide for the operations team.',
            'chosen': 'Alex has completed the first draft of the onboarding guide. The draft is now ready for review.',
            'rejected': 'Has the first draft of the onboarding guide been finished yet?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please fix this Python traceback from my Maya project.',
            'chosen': 'Can you paste the traceback and the code that triggers it in the Maya project?',
            'rejected': 'Assuming the Maya project fails on a missing import, install the referenced package in the active environment and rerun the script.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python function named double_17 that returns twice its numeric argument.',
            'chosen': '```python\ndef double_17(value):\n    return value * 2\n```',
            'rejected': 'What should the function double_17 do to its argument?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 84% of 200?',
            'chosen': '168.',
            'rejected': 'What percentage should I calculate from 200?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python function named double_8 that returns twice its numeric argument.',
            'chosen': '```python\ndef double_8(value):\n    return value * 2\n```',
            'rejected': 'What should the function double_8 do to its argument?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Find the bug in the SQL query for Fatima's dashboard.",
            'chosen': "Can you paste the SQL query and describe the wrong result or error in Fatima's dashboard?",
            'rejected': 'Assuming the query duplicates rows after a join, aggregate or deduplicate the joined table before calculating dashboard totals.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a one-sentence description of a folder named “Onboarding Guide” that stores drafts and feedback for the research group.',
            'chosen': 'This folder contains drafts of the onboarding guide and the feedback collected during review.',
            'rejected': 'What kind of files will the folder contain?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python function named double_4 that returns twice its numeric argument.',
            'chosen': '```python\ndef double_4(value):\n    return value * 2\n```',
            'rejected': 'What should the function double_4 do to its argument?',
        },

        {
            'class': 'clear',
            'prompt': 'Give a one-sentence definition of a scanner warranty.',
            'chosen': 'A scanner warranty is a promise by a seller or manufacturer to repair, replace, or otherwise remedy certain covered problems under stated conditions.',
            'rejected': 'Which product’s warranty would you like defined?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Find the bug in the SQL query for Drew's dashboard.",
            'chosen': "Can you paste the SQL query and describe the wrong result or error in Drew's dashboard?",
            'rejected': 'Assuming the query duplicates rows after a join, aggregate or deduplicate the joined table before calculating dashboard totals.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Help me respond to the complaint in Theo's training handout for the design review.",
            'chosen': 'What does the complaint in the training handout say?',
            'rejected': 'Assuming the complaint concerns a delay, reply: “I apologize for the delay. We are reviewing the issue and will follow up with an update.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence thank-you message to Elena for helping with a camera setup.',
            'chosen': 'Thanks so much, Elena, for helping me set up my camera—I really appreciate it!',
            'rejected': 'Who should I thank for the camera setup?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 19 plus 11? Give just the number.',
            'chosen': '30',
            'rejected': 'What two numbers should I add for this calculation of 19 plus 11?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Check whether the printer warranty covers my problem.',
            'chosen': 'Which printer warranty terms apply, and what happened to the device?',
            'rejected': 'Assuming the printer has a standard one-year warranty and a manufacturing defect, it may be covered; accidental damage is commonly excluded.',
        },

        {
            'class': 'clear',
            'prompt': 'Explain in one sentence what a Python for loop does, using the variable avery as an example.',
            'chosen': 'A Python `for` loop iterates over items in a sequence, assigning each item to a variable such as `avery` in turn.',
            'rejected': 'What programming language should I use to explain the loop?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence thank-you message to Drew for helping with a microphone setup.',
            'chosen': 'Thanks so much, Drew, for helping me set up my microphone—I really appreciate it!',
            'rejected': 'Who should I thank for the microphone setup?',
        },

        {
            'class': 'clear',
            'prompt': 'Explain in one sentence what a Python for loop does, using the variable alex as an example.',
            'chosen': 'A Python `for` loop iterates over items in a sequence, assigning each item to a variable such as `alex` in turn.',
            'rejected': 'What programming language should I use to explain the loop?',
        },

        {
            'class': 'clear',
            'prompt': 'Sort these three words alphabetically: riley, apple, zebra.',
            'chosen': 'apple, riley, zebra.',
            'rejected': 'Which words should I alphabetize along with riley?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What is the next deadline for the Theo application?',
            'chosen': 'Which Theo application or program are you referring to?',
            'rejected': 'Assuming the Theo application is for a program with a November 1 deadline, plan to submit by October 31; verify the actual deadline on the program’s site.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence thank-you message to Riley for helping with a router setup.',
            'chosen': 'Thanks so much, Riley, for helping me set up my router—I really appreciate it!',
            'rejected': 'Who should I thank for the router setup?',
        },

        {
            'class': 'clear',
            'prompt': 'Give one clear call-to-action sentence for a travel itinerary asking readers to send comments by Monday for the community team.',
            'chosen': 'Please send your comments on the travel itinerary by Monday.',
            'rejected': 'When should readers send their comments?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Edit the paragraph Noah wrote for clarity.',
            'chosen': 'Can you paste the paragraph written by Noah?',
            'rejected': 'Assuming Noah wrote “Due to the fact that we were late, we missed it,” a clearer version is “Because we were late, we missed it.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Edit the paragraph Omar wrote for clarity.',
            'chosen': 'Can you paste the paragraph written by Omar?',
            'rejected': 'Assuming Omar wrote “Due to the fact that we were late, we missed it,” a clearer version is “Because we were late, we missed it.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence thank-you message to Jordan for helping with a phone setup.',
            'chosen': 'Thanks so much, Jordan, for helping me set up my phone—I really appreciate it!',
            'rejected': 'Who should I thank for the phone setup?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 21 multiplied by 13?',
            'chosen': '273',
            'rejected': 'What multiplier should I use with 21?',
        },

        {
            'class': 'clear',
            'prompt': 'Give one clear call-to-action sentence for a product description asking readers to send comments by Monday for the student organization.',
            'chosen': 'Please send your comments on the product description by Monday.',
            'rejected': 'When should readers send their comments?',
        },

        {
            'class': 'clear',
            'prompt': 'Give a concise title for a meeting where Quinn reviews the class presentation with the team for the research group.',
            'chosen': 'Class Presentation Review with Quinn',
            'rejected': 'What document is Quinn reviewing in the meeting?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Create a reply to the question in Drew's customer FAQ for the volunteer group.",
            'chosen': 'Can you share the question from the customer FAQ and any relevant context?',
            'rejected': 'Assuming they asked for a status update, reply: “The work is in progress, and I will share the next update once it is ready.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Write a reply to the email Maya sent me yesterday.',
            'chosen': 'Can you share the email from Maya and what you want your reply to accomplish?',
            'rejected': 'Assuming Maya asked for a status update, you could reply: “Hi Maya, thanks for checking in. I’m making progress and will send an update soon.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python function named double_6 that returns twice its numeric argument.',
            'chosen': '```python\ndef double_6(value):\n    return value * 2\n```',
            'rejected': 'What should the function double_6 do to its argument?',
        },

        {
            'class': 'clear',
            'prompt': 'Explain in one sentence what a Python for loop does, using the variable zoe as an example.',
            'chosen': 'A Python `for` loop iterates over items in a sequence, assigning each item to a variable such as `zoe` in turn.',
            'rejected': 'What programming language should I use to explain the loop?',
        },

        {
            'class': 'clear',
            'prompt': 'Rewrite “Lucas was late because the bus broke down” in the present tense.',
            'chosen': 'Lucas is late because the bus breaks down.',
            'rejected': 'What sentence about Lucas should I rewrite?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a brief, neutral announcement that Maya has finished the first draft of the support article for the volunteer group.',
            'chosen': 'Maya has completed the first draft of the support article. The draft is now ready for review.',
            'rejected': 'Has the first draft of the support article been finished yet?',
        },

        {
            'class': 'clear',
            'prompt': 'Give a concise title for a meeting where Riley reviews the workshop outline with the team for the student organization.',
            'chosen': 'Workshop Outline Review with Riley',
            'rejected': 'What document is Riley reviewing in the meeting?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Graph the sales numbers for the microphone launch.',
            'chosen': 'Can you provide the sales figures and the dates or categories for the microphone launch?',
            'rejected': 'Assuming microphone sales were 20, 35, and 50 units over three weeks, a line chart would show steady growth from week one to week three.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a short message telling Morgan that the release notes will be discussed at tomorrow’s meeting for the campus club.',
            'chosen': 'Hi Morgan, we’ll discuss the release notes at tomorrow’s meeting.',
            'rejected': 'When will the release notes be discussed?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'How many miles did Omar travel on the trip?',
            'chosen': 'What was the trip’s starting point and destination, or the recorded distance?',
            'rejected': 'Assuming Omar traveled 40 miles outbound and 40 miles back, the total distance was 80 miles.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you compare the two printer models on my shortlist?',
            'chosen': 'Which two printer models are you comparing?',
            'rejected': 'Assuming your shortlist is the basic and premium printer, the premium model generally offers more features while the basic model costs less.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Create a calendar invite for my meeting with Drew.',
            'chosen': 'What date, start time, time zone, and duration should the meeting with Drew have?',
            'rejected': 'Assuming the meeting with Drew is tomorrow from 2:00 to 2:30 PM local time, use the title “Catch-up with Drew” and add those details to a calendar invitation.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite one-sentence thank-you to Noah for reviewing the application letter for the customer support team.',
            'chosen': 'Thank you, Noah, for taking the time to review the application letter—I appreciate your feedback.',
            'rejected': 'Who should I thank for reviewing the application letter?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a two-item checklist for publishing the website homepage after Drew approves it for the campus club.',
            'chosen': '1. Confirm Drew has approved the final version. 2. Publish the approved website homepage in the intended channel.',
            'rejected': 'Whose approval is needed before publishing the website homepage?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a one-sentence description of a folder named “Onboarding Guide” that stores drafts and feedback for the campus club.',
            'chosen': 'This folder contains drafts of the onboarding guide and the feedback collected during review.',
            'rejected': 'What kind of files will the folder contain?',
        },

        {
            'class': 'clear',
            'prompt': 'Rewrite “Zoe was late because the bus broke down” in the present tense.',
            'chosen': 'Zoe is late because the bus breaks down.',
            'rejected': 'What sentence about Zoe should I rewrite?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a two-sentence opening for a presentation by Leah introducing a customer FAQ for the community team.',
            'chosen': 'Hello everyone, I’m Leah, and today I’ll introduce our customer FAQ. I’ll cover its purpose and the key points to remember.',
            'rejected': 'Who will be presenting the customer FAQ?',
        },

        {
            'class': 'clear',
            'prompt': 'Give a one-sentence definition of a e-reader warranty.',
            'chosen': 'A e-reader warranty is a promise by a seller or manufacturer to repair, replace, or otherwise remedy certain covered problems under stated conditions.',
            'rejected': 'Which product’s warranty would you like defined?',
        },

        {
            'class': 'clear',
            'prompt': 'Sort these three words alphabetically: omar, apple, zebra.',
            'chosen': 'apple, omar, zebra.',
            'rejected': 'Which words should I alphabetize along with omar?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Fill in the missing cells in my Alex expense spreadsheet.',
            'chosen': 'Can you upload the Alex spreadsheet and specify which missing cells to fill?',
            'rejected': 'Assuming the missing Alex cells are monthly totals, fill each one with a SUM formula over that month’s expense entries.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence reminder to Jordan to review the project proposal by Friday for the campus club.',
            'chosen': 'Hi Jordan, just a friendly reminder to review the project proposal by Friday—thanks!',
            'rejected': 'What deadline should I mention in the reminder to Jordan?',
        },

        {
            'class': 'clear',
            'prompt': 'Give a one-sentence definition of a tablet warranty.',
            'chosen': 'A tablet warranty is a promise by a seller or manufacturer to repair, replace, or otherwise remedy certain covered problems under stated conditions.',
            'rejected': 'Which product’s warranty would you like defined?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What does this error code on my tablet mean?',
            'chosen': 'What exact error code is displayed on your tablet, and what is its make and model?',
            'rejected': 'Assuming your tablet shows an overheating warning, turn it off, let it cool, and check that its vents are unobstructed.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Identify the tone of the release notes written by Noah for the design review.',
            'chosen': 'Could you paste the release notes so I can assess its tone?',
            'rejected': 'Assuming it uses formal greetings and direct requests, its tone is professional and courteous.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Book a table for Maya's celebration.",
            'chosen': "Which restaurant, date, time, and party size should I use for Maya's celebration?",
            'rejected': 'Assuming Maya wants dinner for four at 7 PM on Saturday, look for a restaurant reservation with those details; no booking has been made.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Find the broken link in the volunteer handbook Leah sent for the student organization.',
            'chosen': 'Can you share the volunteer handbook or its URL so I can inspect its links?',
            'rejected': 'Assuming the main resource link is broken, replace it with the current official resource URL and verify it opens.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite follow-up to Jordan asking whether they saw the training handout sent yesterday for the campus club.',
            'chosen': 'Hi Jordan, just checking whether you had a chance to see the training handout I sent yesterday.',
            'rejected': 'When was the training handout sent to Jordan?',
        },

        {
            'class': 'clear',
            'prompt': 'Sort these three words alphabetically: drew, apple, zebra.',
            'chosen': 'apple, drew, zebra.',
            'rejected': 'Which words should I alphabetize along with drew?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a professional email subject line asking Quinn for feedback on the onboarding guide for the volunteer group.',
            'chosen': 'Feedback Requested: Onboarding Guide',
            'rejected': 'What is the purpose of the email to Quinn?',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a three-item agenda for a 15-minute meeting with Sam about the release notes for the campus club.',
            'chosen': '1. Goal and current status (3 min). 2. Key issues and decisions (8 min). 3. Owners and next steps (4 min).',
            'rejected': 'How long is the meeting with Sam supposed to be?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'How many miles did Maya travel on the trip?',
            'chosen': 'What was the trip’s starting point and destination, or the recorded distance?',
            'rejected': 'Assuming Maya traveled 40 miles outbound and 40 miles back, the total distance was 80 miles.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Write a reply to the email Zoe sent me yesterday.',
            'chosen': 'Can you share the email from Zoe and what you want your reply to accomplish?',
            'rejected': 'Assuming Zoe asked for a status update, you could reply: “Hi Zoe, thanks for checking in. I’m making progress and will send an update soon.”',
        },

        {
            'class': 'ambiguous',
            'prompt': "Find the bug in the SQL query for Nina's dashboard.",
            'chosen': "Can you paste the SQL query and describe the wrong result or error in Nina's dashboard?",
            'rejected': 'Assuming the query duplicates rows after a join, aggregate or deduplicate the joined table before calculating dashboard totals.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What does this error code on my microphone mean?',
            'chosen': 'What exact error code is displayed on your microphone, and what is its make and model?',
            'rejected': 'Assuming your microphone shows an overheating warning, turn it off, let it cool, and check that its vents are unobstructed.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please summarize the article Omar sent me.',
            'chosen': 'Can you paste or upload the article Omar sent you?',
            'rejected': 'Assuming Omar sent an article about urban trees, its main idea is that tree cover can reduce heat and improve city living conditions.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Create a calendar invite for my meeting with Theo.',
            'chosen': 'What date, start time, time zone, and duration should the meeting with Theo have?',
            'rejected': 'Assuming the meeting with Theo is tomorrow from 2:00 to 2:30 PM local time, use the title “Catch-up with Theo” and add those details to a calendar invitation.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Answer the question Casey asked in the volunteer handbook for the community team.',
            'chosen': 'What question did Casey ask in the volunteer handbook?',
            'rejected': 'Assuming the question concerns the schedule, a possible answer is: “We expect to share an updated timeline next week.”',
        },

        {
            'class': 'ambiguous',
            'prompt': "Rewrite Priya's support article in a more professional tone for the community team.",
            'chosen': "Can you share the text of Priya's support article to rewrite?",
            'rejected': 'Assuming it announces an update, here is a professional rewrite: “We are pleased to share an update and will provide further details shortly.”',
        },

        {
            'class': 'clear',
            'prompt': 'What is 27 multiplied by 19?',
            'chosen': '513',
            'rejected': 'What multiplier should I use with 27?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 12 multiplied by 4?',
            'chosen': '48',
            'rejected': 'What multiplier should I use with 12?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a professional email subject line asking Theo for feedback on the portfolio bio for the customer support team.',
            'chosen': 'Feedback Requested: Portfolio Bio',
            'rejected': 'What is the purpose of the email to Theo?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a short message telling Morgan that the release notes will be discussed at tomorrow’s meeting for the research group.',
            'chosen': 'Hi Morgan, we’ll discuss the release notes at tomorrow’s meeting.',
            'rejected': 'When will the release notes be discussed?',
        },

        {
            'class': 'clear',
            'prompt': 'Give one clear call-to-action sentence for a team newsletter asking readers to send comments by Monday for the campus club.',
            'chosen': 'Please send your comments on the team newsletter by Monday.',
            'rejected': 'When should readers send their comments?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Compare the project proposal from Fatima with the earlier draft for the customer support team.',
            'chosen': 'Can you share both the current project proposal and the earlier draft?',
            'rejected': 'Assuming the new version adds details, it is more informative but may benefit from trimming.',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a SQL query to select the name column from a table called orders_13.',
            'chosen': '```sql\nSELECT name FROM orders_13;\n```',
            'rejected': 'Which table should the SQL query read from?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you compare the two thermostat models on my shortlist?',
            'chosen': 'Which two thermostat models are you comparing?',
            'rejected': 'Assuming your shortlist is the basic and premium thermostat, the premium model generally offers more features while the basic model costs less.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What is the next deadline for the Jordan application?',
            'chosen': 'Which Jordan application or program are you referring to?',
            'rejected': 'Assuming the Jordan application is for a program with a November 1 deadline, plan to submit by October 31; verify the actual deadline on the program’s site.',
        },

        {
            'class': 'clear',
            'prompt': 'Give a one-sentence definition of a game console warranty.',
            'chosen': 'A game console warranty is a promise by a seller or manufacturer to repair, replace, or otherwise remedy certain covered problems under stated conditions.',
            'rejected': 'Which product’s warranty would you like defined?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you tell me whether my keyboard is compatible with the accessory I found?',
            'chosen': 'What is the exact keyboard model and the accessory model?',
            'rejected': 'Assuming both the keyboard and accessory use standard USB-C data connections, they may work together; check the specifications before purchasing.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence thank-you message to Theo for helping with a thermostat setup.',
            'chosen': 'Thanks so much, Theo, for helping me set up my thermostat—I really appreciate it!',
            'rejected': 'Who should I thank for the thermostat setup?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence reminder to Iris to review the support article by Friday for the student organization.',
            'chosen': 'Hi Iris, just a friendly reminder to review the support article by Friday—thanks!',
            'rejected': 'What deadline should I mention in the reminder to Iris?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python function named double_2 that returns twice its numeric argument.',
            'chosen': '```python\ndef double_2(value):\n    return value * 2\n```',
            'rejected': 'What should the function double_2 do to its argument?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Graph the sales numbers for the router launch.',
            'chosen': 'Can you provide the sales figures and the dates or categories for the router launch?',
            'rejected': 'Assuming router sales were 20, 35, and 50 units over three weeks, a line chart would show steady growth from week one to week three.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please summarize the article Jordan sent me.',
            'chosen': 'Can you paste or upload the article Jordan sent you?',
            'rejected': 'Assuming Jordan sent an article about urban trees, its main idea is that tree cover can reduce heat and improve city living conditions.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Graph the sales numbers for the headphones launch.',
            'chosen': 'Can you provide the sales figures and the dates or categories for the headphones launch?',
            'rejected': 'Assuming headphones sales were 20, 35, and 50 units over three weeks, a line chart would show steady growth from week one to week three.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite follow-up to Maya asking whether they saw the product description sent yesterday for the operations team.',
            'chosen': 'Hi Maya, just checking whether you had a chance to see the product description I sent yesterday.',
            'rejected': 'When was the product description sent to Maya?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please summarize the article Avery sent me.',
            'chosen': 'Can you paste or upload the article Avery sent you?',
            'rejected': 'Assuming Avery sent an article about urban trees, its main idea is that tree cover can reduce heat and improve city living conditions.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Check the factual claims in Theo's training handout for the product team.",
            'chosen': 'Can you provide the training handout or the claims you want checked?',
            'rejected': 'Assuming it claims the project launched last year, I would flag that date for verification against official records.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a one-sentence request asking Maya to share the latest class presentation as a PDF for the design review.',
            'chosen': 'Hi Maya, could you please send me the latest class presentation as a PDF?',
            'rejected': 'What file format should I request from Maya?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence thank-you message to Lucas for helping with a game console setup.',
            'chosen': 'Thanks so much, Lucas, for helping me set up my game console—I really appreciate it!',
            'rejected': 'Who should I thank for the game console setup?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Fill in the missing cells in my Omar expense spreadsheet.',
            'chosen': 'Can you upload the Omar spreadsheet and specify which missing cells to fill?',
            'rejected': 'Assuming the missing Omar cells are monthly totals, fill each one with a SUM formula over that month’s expense entries.',
        },

        {
            'class': 'clear',
            'prompt': 'Give a one-sentence definition of a headphones warranty.',
            'chosen': 'A headphones warranty is a promise by a seller or manufacturer to repair, replace, or otherwise remedy certain covered problems under stated conditions.',
            'rejected': 'Which product’s warranty would you like defined?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a two-sentence opening for a presentation by Rowan introducing a website homepage for the product team.',
            'chosen': 'Hello everyone, I’m Rowan, and today I’ll introduce our website homepage. I’ll cover its purpose and the key points to remember.',
            'rejected': 'Who will be presenting the website homepage?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Find the bug in the SQL query for Eli's dashboard.",
            'chosen': "Can you paste the SQL query and describe the wrong result or error in Eli's dashboard?",
            'rejected': 'Assuming the query duplicates rows after a join, aggregate or deduplicate the joined table before calculating dashboard totals.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Fill in the missing cells in my Priya expense spreadsheet.',
            'chosen': 'Can you upload the Priya spreadsheet and specify which missing cells to fill?',
            'rejected': 'Assuming the missing Priya cells are monthly totals, fill each one with a SUM formula over that month’s expense entries.',
        },

        {
            'class': 'clear',
            'prompt': "Create a simple Markdown heading and two bullet placeholders for Theo's workshop outline for the volunteer group.",
            'chosen': '# Workshop Outline\n- Key point 1\n- Key point 2',
            'rejected': 'What format should I use for the workshop outline?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite one-sentence thank-you to Alex for reviewing the product description for the product team.',
            'chosen': 'Thank you, Alex, for taking the time to review the product description—I appreciate your feedback.',
            'rejected': 'Who should I thank for reviewing the product description?',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a three-item agenda for a 15-minute meeting with Leah about the website homepage for the workshop team.',
            'chosen': '1. Goal and current status (3 min). 2. Key issues and decisions (8 min). 3. Owners and next steps (4 min).',
            'rejected': 'How long is the meeting with Leah supposed to be?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Compare the project proposal from Fatima with the earlier draft for the community team.',
            'chosen': 'Can you share both the current project proposal and the earlier draft?',
            'rejected': 'Assuming the new version adds details, it is more informative but may benefit from trimming.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you tell me whether my e-reader is compatible with the accessory I found?',
            'chosen': 'What is the exact e-reader model and the accessory model?',
            'rejected': 'Assuming both the e-reader and accessory use standard USB-C data connections, they may work together; check the specifications before purchasing.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What does this error code on my smartwatch mean?',
            'chosen': 'What exact error code is displayed on your smartwatch, and what is its make and model?',
            'rejected': 'Assuming your smartwatch shows an overheating warning, turn it off, let it cool, and check that its vents are unobstructed.',
        },

        {
            'class': 'clear',
            'prompt': 'Explain in one sentence what a Python for loop does, using the variable fatima as an example.',
            'chosen': 'A Python `for` loop iterates over items in a sequence, assigning each item to a variable such as `fatima` in turn.',
            'rejected': 'What programming language should I use to explain the loop?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Extract the action items from Sam's meeting agenda for the community team.",
            'chosen': 'Could you share the meeting agenda so I can identify its action items?',
            'rejected': 'Assuming it covers a planning meeting, the action items are to confirm owners, set deadlines, and share the plan.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please summarize the article Riley sent me.',
            'chosen': 'Can you paste or upload the article Riley sent you?',
            'rejected': 'Assuming Riley sent an article about urban trees, its main idea is that tree cover can reduce heat and improve city living conditions.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you compare the two keyboard models on my shortlist?',
            'chosen': 'Which two keyboard models are you comparing?',
            'rejected': 'Assuming your shortlist is the basic and premium keyboard, the premium model generally offers more features while the basic model costs less.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What does this error code on my smart TV mean?',
            'chosen': 'What exact error code is displayed on your smart TV, and what is its make and model?',
            'rejected': 'Assuming your smart TV shows an overheating warning, turn it off, let it cool, and check that its vents are unobstructed.',
        },

        {
            'class': 'clear',
            'prompt': 'Elena got 19 correct answers out of 22. What percentage is that? Round to one decimal.',
            'chosen': '86.4%.',
            'rejected': 'How many questions did Elena answer correctly?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence thank-you message to Omar for helping with a speaker setup.',
            'chosen': 'Thanks so much, Omar, for helping me set up my speaker—I really appreciate it!',
            'rejected': 'Who should I thank for the speaker setup?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a short message telling Nina that the website homepage will be discussed at tomorrow’s meeting for the workshop team.',
            'chosen': 'Hi Nina, we’ll discuss the website homepage at tomorrow’s meeting.',
            'rejected': 'When will the website homepage be discussed?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Put the references from Iris's paper in alphabetical order.",
            'chosen': "Can you paste the references from Iris's paper?",
            'rejected': 'Assuming the authors are Chen, Adams, and Brown, the alphabetical order is Adams, Brown, Chen.',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a SQL query to select the name column from a table called orders_14.',
            'chosen': '```sql\nSELECT name FROM orders_14;\n```',
            'rejected': 'Which table should the SQL query read from?',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a SQL query to select the name column from a table called orders_12.',
            'chosen': '```sql\nSELECT name FROM orders_12;\n```',
            'rejected': 'Which table should the SQL query read from?',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a SQL query to select the name column from a table called orders_5.',
            'chosen': '```sql\nSELECT name FROM orders_5;\n```',
            'rejected': 'Which table should the SQL query read from?',
        },

        {
            'class': 'clear',
            'prompt': 'Give a concise title for a meeting where Eli reviews the meeting agenda with the team for the workshop team.',
            'chosen': 'Meeting Agenda Review with Eli',
            'rejected': 'What document is Eli reviewing in the meeting?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 15 plus 7? Give just the number.',
            'chosen': '22',
            'rejected': 'What two numbers should I add for this calculation of 15 plus 7?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Find the broken link in the workshop outline Fatima sent for the design review.',
            'chosen': 'Can you share the workshop outline or its URL so I can inspect its links?',
            'rejected': 'Assuming the main resource link is broken, replace it with the current official resource URL and verify it opens.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python list containing the integers 1, 2, and 3.',
            'chosen': '`[1, 2, 3]`',
            'rejected': 'Which integers should I put in the Python list?',
        },

        {
            'class': 'clear',
            'prompt': 'Explain in one sentence what a Python for loop does, using the variable eli as an example.',
            'chosen': 'A Python `for` loop iterates over items in a sequence, assigning each item to a variable such as `eli` in turn.',
            'rejected': 'What programming language should I use to explain the loop?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Identify the tone of the club announcement written by Priya for the customer support team.',
            'chosen': 'Could you paste the club announcement so I can assess its tone?',
            'rejected': 'Assuming it uses formal greetings and direct requests, its tone is professional and courteous.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Make a slide from the community poster Sam wrote for the volunteer group.',
            'chosen': 'Can you paste the community poster so I can turn its key points into a slide?',
            'rejected': 'Assuming it describes a project, use a slide titled “Project Overview” with goals, approach, and next steps.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you compare the two smart TV models on my shortlist?',
            'chosen': 'Which two smart TV models are you comparing?',
            'rejected': 'Assuming your shortlist is the basic and premium smart TV, the premium model generally offers more features while the basic model costs less.',
        },

        {
            'class': 'clear',
            'prompt': "Create a simple Markdown heading and two bullet placeholders for Iris's onboarding guide for the product team.",
            'chosen': '# Onboarding Guide\n- Key point 1\n- Key point 2',
            'rejected': 'What format should I use for the onboarding guide?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Check the factual claims in Avery's travel itinerary for the customer support team.",
            'chosen': 'Can you provide the travel itinerary or the claims you want checked?',
            'rejected': 'Assuming it claims the project launched last year, I would flag that date for verification against official records.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Rewrite Alex's class presentation in a more professional tone for the campus club.",
            'chosen': "Can you share the text of Alex's class presentation to rewrite?",
            'rejected': 'Assuming it announces an update, here is a professional rewrite: “We are pleased to share an update and will provide further details shortly.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite subject line for an email to Riley about the router delivery.',
            'chosen': 'Subject: Update on Router Delivery for Riley',
            'rejected': 'Who is the email about the router delivery addressed to?',
        },

        {
            'class': 'clear',
            'prompt': 'Eli got 31 correct answers out of 34. What percentage is that? Round to one decimal.',
            'chosen': '91.2%.',
            'rejected': 'How many questions did Eli answer correctly?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite subject line for an email to Sam about the printer delivery.',
            'chosen': 'Subject: Update on Printer Delivery for Sam',
            'rejected': 'Who is the email about the printer delivery addressed to?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 16 multiplied by 8?',
            'chosen': '128',
            'rejected': 'What multiplier should I use with 16?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What is the next deadline for the Alex application?',
            'chosen': 'Which Alex application or program are you referring to?',
            'rejected': 'Assuming the Alex application is for a program with a November 1 deadline, plan to submit by October 31; verify the actual deadline on the program’s site.',
        },

        {
            'class': 'clear',
            'prompt': 'Sort these three words alphabetically: sam, apple, zebra.',
            'chosen': 'apple, sam, zebra.',
            'rejected': 'Which words should I alphabetize along with sam?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Summarize the project proposal that Jordan shared with me for the operations team.',
            'chosen': 'Could you paste or upload the project proposal you want summarized?',
            'rejected': 'Assuming the project proposal introduces a new initiative, the main points are its goals, timeline, and next steps.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Graph the sales numbers for the fitness tracker launch.',
            'chosen': 'Can you provide the sales figures and the dates or categories for the fitness tracker launch?',
            'rejected': 'Assuming fitness tracker sales were 20, 35, and 50 units over three weeks, a line chart would show steady growth from week one to week three.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Fill in the missing cells in my Lucas expense spreadsheet.',
            'chosen': 'Can you upload the Lucas spreadsheet and specify which missing cells to fill?',
            'rejected': 'Assuming the missing Lucas cells are monthly totals, fill each one with a SUM formula over that month’s expense entries.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What does this error code on my keyboard mean?',
            'chosen': 'What exact error code is displayed on your keyboard, and what is its make and model?',
            'rejected': 'Assuming your keyboard shows an overheating warning, turn it off, let it cool, and check that its vents are unobstructed.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Edit the paragraph Nina wrote for clarity.',
            'chosen': 'Can you paste the paragraph written by Nina?',
            'rejected': 'Assuming Nina wrote “Due to the fact that we were late, we missed it,” a clearer version is “Because we were late, we missed it.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence reminder to Fatima to review the portfolio bio by Friday for the product team.',
            'chosen': 'Hi Fatima, just a friendly reminder to review the portfolio bio by Friday—thanks!',
            'rejected': 'What deadline should I mention in the reminder to Fatima?',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a three-item agenda for a 15-minute meeting with Fatima about the training handout for the campus club.',
            'chosen': '1. Goal and current status (3 min). 2. Key issues and decisions (8 min). 3. Owners and next steps (4 min).',
            'rejected': 'How long is the meeting with Fatima supposed to be?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence thank-you message to Maya for helping with a laptop setup.',
            'chosen': 'Thanks so much, Maya, for helping me set up my laptop—I really appreciate it!',
            'rejected': 'Who should I thank for the laptop setup?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Fill in the missing cells in my Iris expense spreadsheet.',
            'chosen': 'Can you upload the Iris spreadsheet and specify which missing cells to fill?',
            'rejected': 'Assuming the missing Iris cells are monthly totals, fill each one with a SUM formula over that month’s expense entries.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a JSON object with key "owner" set to "Fatima" and key "quantity" set to 17.',
            'chosen': '`{"owner": "Fatima", "quantity": 17}`',
            'rejected': 'What owner name should I put in the JSON object?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a short message telling Iris that the onboarding guide will be discussed at tomorrow’s meeting for the design review.',
            'chosen': 'Hi Iris, we’ll discuss the onboarding guide at tomorrow’s meeting.',
            'rejected': 'When will the onboarding guide be discussed?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 83% of 200?',
            'chosen': '166.',
            'rejected': 'What percentage should I calculate from 200?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Translate the note I wrote for Elena into Spanish.',
            'chosen': 'What is the text of the note you want translated for Elena?',
            'rejected': 'Assuming your note says “See you tomorrow,” the Spanish translation is “Nos vemos mañana.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Create a calendar invite for my meeting with Jordan.',
            'chosen': 'What date, start time, time zone, and duration should the meeting with Jordan have?',
            'rejected': 'Assuming the meeting with Jordan is tomorrow from 2:00 to 2:30 PM local time, use the title “Catch-up with Jordan” and add those details to a calendar invitation.',
        },

        {
            'class': 'clear',
            'prompt': 'Give a one-sentence definition of a desktop warranty.',
            'chosen': 'A desktop warranty is a promise by a seller or manufacturer to repair, replace, or otherwise remedy certain covered problems under stated conditions.',
            'rejected': 'Which product’s warranty would you like defined?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Choose which version of Eli's team newsletter is better for the community team.",
            'chosen': 'Can you share both versions of the team newsletter so I can compare them?',
            'rejected': 'Assuming one version is shorter and clearer, I would choose that one because it is easier to scan.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please fix this Python traceback from my Elena project.',
            'chosen': 'Can you paste the traceback and the code that triggers it in the Elena project?',
            'rejected': 'Assuming the Elena project fails on a missing import, install the referenced package in the active environment and rerun the script.',
        },

        {
            'class': 'clear',
            'prompt': 'Give a concise title for a meeting where Leah reviews the event flyer with the team for the volunteer group.',
            'chosen': 'Event Flyer Review with Leah',
            'rejected': 'What document is Leah reviewing in the meeting?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Check whether the smartwatch warranty covers my problem.',
            'chosen': 'Which smartwatch warranty terms apply, and what happened to the device?',
            'rejected': 'Assuming the smartwatch has a standard one-year warranty and a manufacturing defect, it may be covered; accidental damage is commonly excluded.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What does this error code on my thermostat mean?',
            'chosen': 'What exact error code is displayed on your thermostat, and what is its make and model?',
            'rejected': 'Assuming your thermostat shows an overheating warning, turn it off, let it cool, and check that its vents are unobstructed.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite one-sentence thank-you to Casey for reviewing the volunteer handbook for the community team.',
            'chosen': 'Thank you, Casey, for taking the time to review the volunteer handbook—I appreciate your feedback.',
            'rejected': 'Who should I thank for reviewing the volunteer handbook?',
        },

        {
            'class': 'clear',
            'prompt': 'Give a one-sentence definition of a smartwatch warranty.',
            'chosen': 'A smartwatch warranty is a promise by a seller or manufacturer to repair, replace, or otherwise remedy certain covered problems under stated conditions.',
            'rejected': 'Which product’s warranty would you like defined?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'How many miles did Nina travel on the trip?',
            'chosen': 'What was the trip’s starting point and destination, or the recorded distance?',
            'rejected': 'Assuming Nina traveled 40 miles outbound and 40 miles back, the total distance was 80 miles.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Create a calendar invite for my meeting with Omar.',
            'chosen': 'What date, start time, time zone, and duration should the meeting with Omar have?',
            'rejected': 'Assuming the meeting with Omar is tomorrow from 2:00 to 2:30 PM local time, use the title “Catch-up with Omar” and add those details to a calendar invitation.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Edit the paragraph Leah wrote for clarity.',
            'chosen': 'Can you paste the paragraph written by Leah?',
            'rejected': 'Assuming Leah wrote “Due to the fact that we were late, we missed it,” a clearer version is “Because we were late, we missed it.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a concise checklist with three steps for proofreading a application letter for the volunteer group.',
            'chosen': '1. Check spelling and grammar. 2. Verify names and facts. 3. Read once for clarity and consistency.',
            'rejected': 'How many proofreading steps would you like for the application letter?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a concise checklist with three steps for proofreading a volunteer handbook for the student organization.',
            'chosen': '1. Check spelling and grammar. 2. Verify names and facts. 3. Read once for clarity and consistency.',
            'rejected': 'How many proofreading steps would you like for the volunteer handbook?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Translate the note I wrote for Iris into Spanish.',
            'chosen': 'What is the text of the note you want translated for Iris?',
            'rejected': 'Assuming your note says “See you tomorrow,” the Spanish translation is “Nos vemos mañana.”',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a SQL query to select the name column from a table called orders_0.',
            'chosen': '```sql\nSELECT name FROM orders_0;\n```',
            'rejected': 'Which table should the SQL query read from?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you tell me whether my microphone is compatible with the accessory I found?',
            'chosen': 'What is the exact microphone model and the accessory model?',
            'rejected': 'Assuming both the microphone and accessory use standard USB-C data connections, they may work together; check the specifications before purchasing.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Find the bug in the SQL query for Theo's dashboard.",
            'chosen': "Can you paste the SQL query and describe the wrong result or error in Theo's dashboard?",
            'rejected': 'Assuming the query duplicates rows after a join, aggregate or deduplicate the joined table before calculating dashboard totals.',
        },

        {
            'class': 'clear',
            'prompt': 'Explain in one sentence what a Python for loop does, using the variable elena as an example.',
            'chosen': 'A Python `for` loop iterates over items in a sequence, assigning each item to a variable such as `elena` in turn.',
            'rejected': 'What programming language should I use to explain the loop?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Help me respond to the complaint in Quinn's class presentation for the campus club.",
            'chosen': 'What does the complaint in the class presentation say?',
            'rejected': 'Assuming the complaint concerns a delay, reply: “I apologize for the delay. We are reviewing the issue and will follow up with an update.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a professional email subject line asking Noah for feedback on the meeting agenda for the design review.',
            'chosen': 'Feedback Requested: Meeting Agenda',
            'rejected': 'What is the purpose of the email to Noah?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Turn the product description Maya mentioned into a checklist for the operations team.',
            'chosen': 'Can you share the content or requirements of the product description?',
            'rejected': 'Assuming it is a project kickoff document, use this checklist: confirm scope, assign owners, and agree on deadlines.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please fix this Python traceback from my Sam project.',
            'chosen': 'Can you paste the traceback and the code that triggers it in the Sam project?',
            'rejected': 'Assuming the Sam project fails on a missing import, install the referenced package in the active environment and rerun the script.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What does this error code on my speaker mean?',
            'chosen': 'What exact error code is displayed on your speaker, and what is its make and model?',
            'rejected': 'Assuming your speaker shows an overheating warning, turn it off, let it cool, and check that its vents are unobstructed.',
        },

        {
            'class': 'clear',
            'prompt': 'Rewrite “Elena was late because the bus broke down” in the present tense.',
            'chosen': 'Elena is late because the bus breaks down.',
            'rejected': 'What sentence about Elena should I rewrite?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence reminder to Rowan to review the interview notes by Friday for the workshop team.',
            'chosen': 'Hi Rowan, just a friendly reminder to review the interview notes by Friday—thanks!',
            'rejected': 'What deadline should I mention in the reminder to Rowan?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a brief, neutral announcement that Nina has finished the first draft of the interview notes for the design review.',
            'chosen': 'Nina has completed the first draft of the interview notes. The draft is now ready for review.',
            'rejected': 'Has the first draft of the interview notes been finished yet?',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a three-item agenda for a 15-minute meeting with Eli about the team newsletter for the design review.',
            'chosen': '1. Goal and current status (3 min). 2. Key issues and decisions (8 min). 3. Owners and next steps (4 min).',
            'rejected': 'How long is the meeting with Eli supposed to be?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Write a reply to the email Riley sent me yesterday.',
            'chosen': 'Can you share the email from Riley and what you want your reply to accomplish?',
            'rejected': 'Assuming Riley asked for a status update, you could reply: “Hi Riley, thanks for checking in. I’m making progress and will send an update soon.”',
        },

        {
            'class': 'clear',
            'prompt': 'Alex got 27 correct answers out of 30. What percentage is that? Round to one decimal.',
            'chosen': '90.0%.',
            'rejected': 'How many questions did Alex answer correctly?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please fix this Python traceback from my Avery project.',
            'chosen': 'Can you paste the traceback and the code that triggers it in the Avery project?',
            'rejected': 'Assuming the Avery project fails on a missing import, install the referenced package in the active environment and rerun the script.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a short message telling Maya that the club announcement will be discussed at tomorrow’s meeting for the student organization.',
            'chosen': 'Hi Maya, we’ll discuss the club announcement at tomorrow’s meeting.',
            'rejected': 'When will the club announcement be discussed?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Write a reply to the email Kai sent me yesterday.',
            'chosen': 'Can you share the email from Kai and what you want your reply to accomplish?',
            'rejected': 'Assuming Kai asked for a status update, you could reply: “Hi Kai, thanks for checking in. I’m making progress and will send an update soon.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please fix this Python traceback from my Lucas project.',
            'chosen': 'Can you paste the traceback and the code that triggers it in the Lucas project?',
            'rejected': 'Assuming the Lucas project fails on a missing import, install the referenced package in the active environment and rerun the script.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'How many miles did Lucas travel on the trip?',
            'chosen': 'What was the trip’s starting point and destination, or the recorded distance?',
            'rejected': 'Assuming Lucas traveled 40 miles outbound and 40 miles back, the total distance was 80 miles.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a one-sentence request asking Rowan to share the latest volunteer handbook as a PDF for the volunteer group.',
            'chosen': 'Hi Rowan, could you please send me the latest volunteer handbook as a PDF?',
            'rejected': 'What file format should I request from Rowan?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Summarize the onboarding guide that Maya shared with me for the customer support team.',
            'chosen': 'Could you paste or upload the onboarding guide you want summarized?',
            'rejected': 'Assuming the onboarding guide introduces a new initiative, the main points are its goals, timeline, and next steps.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a one-sentence description of a folder named “Community Poster” that stores drafts and feedback for the operations team.',
            'chosen': 'This folder contains drafts of the community poster and the feedback collected during review.',
            'rejected': 'What kind of files will the folder contain?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please summarize the article Priya sent me.',
            'chosen': 'Can you paste or upload the article Priya sent you?',
            'rejected': 'Assuming Priya sent an article about urban trees, its main idea is that tree cover can reduce heat and improve city living conditions.',
        },

        {
            'class': 'clear',
            'prompt': 'Rewrite “Theo was late because the bus broke down” in the present tense.',
            'chosen': 'Theo is late because the bus breaks down.',
            'rejected': 'What sentence about Theo should I rewrite?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Put the references from Riley's paper in alphabetical order.",
            'chosen': "Can you paste the references from Riley's paper?",
            'rejected': 'Assuming the authors are Chen, Adams, and Brown, the alphabetical order is Adams, Brown, Chen.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite subject line for an email to Eli about the desktop delivery.',
            'chosen': 'Subject: Update on Desktop Delivery for Eli',
            'rejected': 'Who is the email about the desktop delivery addressed to?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Put the references from Drew's paper in alphabetical order.",
            'chosen': "Can you paste the references from Drew's paper?",
            'rejected': 'Assuming the authors are Chen, Adams, and Brown, the alphabetical order is Adams, Brown, Chen.',
        },

        {
            'class': 'clear',
            'prompt': 'Rewrite “Avery was late because the bus broke down” in the present tense.',
            'chosen': 'Avery is late because the bus breaks down.',
            'rejected': 'What sentence about Avery should I rewrite?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Graph the sales numbers for the printer launch.',
            'chosen': 'Can you provide the sales figures and the dates or categories for the printer launch?',
            'rejected': 'Assuming printer sales were 20, 35, and 50 units over three weeks, a line chart would show steady growth from week one to week three.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please fix this Python traceback from my Priya project.',
            'chosen': 'Can you paste the traceback and the code that triggers it in the Priya project?',
            'rejected': 'Assuming the Priya project fails on a missing import, install the referenced package in the active environment and rerun the script.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Create a reply to the question in Noah's community poster for the operations team.",
            'chosen': 'Can you share the question from the community poster and any relevant context?',
            'rejected': 'Assuming they asked for a status update, reply: “The work is in progress, and I will share the next update once it is ready.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you tell me whether my printer is compatible with the accessory I found?',
            'chosen': 'What is the exact printer model and the accessory model?',
            'rejected': 'Assuming both the printer and accessory use standard USB-C data connections, they may work together; check the specifications before purchasing.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a concise checklist with three steps for proofreading a workshop outline for the design review.',
            'chosen': '1. Check spelling and grammar. 2. Verify names and facts. 3. Read once for clarity and consistency.',
            'rejected': 'How many proofreading steps would you like for the workshop outline?',
        },

        {
            'class': 'clear',
            'prompt': "Create a simple Markdown heading and two bullet placeholders for Quinn's support article for the community team.",
            'chosen': '# Support Article\n- Key point 1\n- Key point 2',
            'rejected': 'What format should I use for the support article?',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a SQL query to select the name column from a table called orders_19.',
            'chosen': '```sql\nSELECT name FROM orders_19;\n```',
            'rejected': 'Which table should the SQL query read from?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 18 multiplied by 10?',
            'chosen': '180',
            'rejected': 'What multiplier should I use with 18?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a short message telling Fatima that the project proposal will be discussed at tomorrow’s meeting for the customer support team.',
            'chosen': 'Hi Fatima, we’ll discuss the project proposal at tomorrow’s meeting.',
            'rejected': 'When will the project proposal be discussed?',
        },

        {
            'class': 'clear',
            'prompt': 'Rewrite “Omar was late because the bus broke down” in the present tense.',
            'chosen': 'Omar is late because the bus breaks down.',
            'rejected': 'What sentence about Omar should I rewrite?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python function named double_11 that returns twice its numeric argument.',
            'chosen': '```python\ndef double_11(value):\n    return value * 2\n```',
            'rejected': 'What should the function double_11 do to its argument?',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a SQL query to select the name column from a table called orders_16.',
            'chosen': '```sql\nSELECT name FROM orders_16;\n```',
            'rejected': 'Which table should the SQL query read from?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Book a table for Leah's celebration.",
            'chosen': "Which restaurant, date, time, and party size should I use for Leah's celebration?",
            'rejected': 'Assuming Leah wants dinner for four at 7 PM on Saturday, look for a restaurant reservation with those details; no booking has been made.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence thank-you message to Fatima for helping with a fitness tracker setup.',
            'chosen': 'Thanks so much, Fatima, for helping me set up my fitness tracker—I really appreciate it!',
            'rejected': 'Who should I thank for the fitness tracker setup?',
        },

        {
            'class': 'clear',
            'prompt': 'Explain in one sentence what a Python for loop does, using the variable noah as an example.',
            'chosen': 'A Python `for` loop iterates over items in a sequence, assigning each item to a variable such as `noah` in turn.',
            'rejected': 'What programming language should I use to explain the loop?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python function named double_12 that returns twice its numeric argument.',
            'chosen': '```python\ndef double_12(value):\n    return value * 2\n```',
            'rejected': 'What should the function double_12 do to its argument?',
        },

        {
            'class': 'clear',
            'prompt': 'Give a concise title for a meeting where Riley reviews the workshop outline with the team for the volunteer group.',
            'chosen': 'Workshop Outline Review with Riley',
            'rejected': 'What document is Riley reviewing in the meeting?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a brief, neutral announcement that Kai has finished the first draft of the community poster for the customer support team.',
            'chosen': 'Kai has completed the first draft of the community poster. The draft is now ready for review.',
            'rejected': 'Has the first draft of the community poster been finished yet?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a one-sentence request asking Alex to share the latest support article as a PDF for the student organization.',
            'chosen': 'Hi Alex, could you please send me the latest support article as a PDF?',
            'rejected': 'What file format should I request from Alex?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Edit the paragraph Eli wrote for clarity.',
            'chosen': 'Can you paste the paragraph written by Eli?',
            'rejected': 'Assuming Eli wrote “Due to the fact that we were late, we missed it,” a clearer version is “Because we were late, we missed it.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite follow-up to Nina asking whether they saw the volunteer handbook sent yesterday for the volunteer group.',
            'chosen': 'Hi Nina, just checking whether you had a chance to see the volunteer handbook I sent yesterday.',
            'rejected': 'When was the volunteer handbook sent to Nina?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'How many miles did Fatima travel on the trip?',
            'chosen': 'What was the trip’s starting point and destination, or the recorded distance?',
            'rejected': 'Assuming Fatima traveled 40 miles outbound and 40 miles back, the total distance was 80 miles.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a two-sentence opening for a presentation by Fatima introducing a travel itinerary for the student organization.',
            'chosen': 'Hello everyone, I’m Fatima, and today I’ll introduce our travel itinerary. I’ll cover its purpose and the key points to remember.',
            'rejected': 'Who will be presenting the travel itinerary?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a two-item checklist for publishing the website homepage after Drew approves it for the research group.',
            'chosen': '1. Confirm Drew has approved the final version. 2. Publish the approved website homepage in the intended channel.',
            'rejected': 'Whose approval is needed before publishing the website homepage?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What does this error code on my phone mean?',
            'chosen': 'What exact error code is displayed on your phone, and what is its make and model?',
            'rejected': 'Assuming your phone shows an overheating warning, turn it off, let it cool, and check that its vents are unobstructed.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python function named double_10 that returns twice its numeric argument.',
            'chosen': '```python\ndef double_10(value):\n    return value * 2\n```',
            'rejected': 'What should the function double_10 do to its argument?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Fill in the missing cells in my Leah expense spreadsheet.',
            'chosen': 'Can you upload the Leah spreadsheet and specify which missing cells to fill?',
            'rejected': 'Assuming the missing Leah cells are monthly totals, fill each one with a SUM formula over that month’s expense entries.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a JSON object with key "owner" set to "Sam" and key "quantity" set to 4.',
            'chosen': '`{"owner": "Sam", "quantity": 4}`',
            'rejected': 'What owner name should I put in the JSON object?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a JSON object with key "owner" set to "Leah" and key "quantity" set to 11.',
            'chosen': '`{"owner": "Leah", "quantity": 11}`',
            'rejected': 'What owner name should I put in the JSON object?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Choose which version of Quinn's club announcement is better for the student organization.",
            'chosen': 'Can you share both versions of the club announcement so I can compare them?',
            'rejected': 'Assuming one version is shorter and clearer, I would choose that one because it is easier to scan.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Answer the question Nina asked in the website homepage for the campus club.',
            'chosen': 'What question did Nina ask in the website homepage?',
            'rejected': 'Assuming the question concerns the schedule, a possible answer is: “We expect to share an updated timeline next week.”',
        },

        {
            'class': 'ambiguous',
            'prompt': "Translate Kai's meeting agenda into the language requested by the recipient for the campus club.",
            'chosen': 'Which language did the recipient request, and can you share the meeting agenda text?',
            'rejected': 'Assuming the recipient wants French and the message is a greeting, a translation would be “Bonjour, merci pour votre message.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a one-sentence description of a folder named “Portfolio Bio” that stores drafts and feedback for the design review.',
            'chosen': 'This folder contains drafts of the portfolio bio and the feedback collected during review.',
            'rejected': 'What kind of files will the folder contain?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a one-sentence request asking Maya to share the latest class presentation as a PDF for the community team.',
            'chosen': 'Hi Maya, could you please send me the latest class presentation as a PDF?',
            'rejected': 'What file format should I request from Maya?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a two-sentence opening for a presentation by Iris introducing a club announcement for the campus club.',
            'chosen': 'Hello everyone, I’m Iris, and today I’ll introduce our club announcement. I’ll cover its purpose and the key points to remember.',
            'rejected': 'Who will be presenting the club announcement?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a two-item checklist for publishing the team newsletter after Sam approves it for the volunteer group.',
            'chosen': '1. Confirm Sam has approved the final version. 2. Publish the approved team newsletter in the intended channel.',
            'rejected': 'Whose approval is needed before publishing the team newsletter?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Edit the paragraph Zoe wrote for clarity.',
            'chosen': 'Can you paste the paragraph written by Zoe?',
            'rejected': 'Assuming Zoe wrote “Due to the fact that we were late, we missed it,” a clearer version is “Because we were late, we missed it.”',
        },

        {
            'class': 'clear',
            'prompt': 'Rewrite “Noah was late because the bus broke down” in the present tense.',
            'chosen': 'Noah is late because the bus breaks down.',
            'rejected': 'What sentence about Noah should I rewrite?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Put the references from Eli's paper in alphabetical order.",
            'chosen': "Can you paste the references from Eli's paper?",
            'rejected': 'Assuming the authors are Chen, Adams, and Brown, the alphabetical order is Adams, Brown, Chen.',
        },

        {
            'class': 'clear',
            'prompt': 'Leah got 22 correct answers out of 25. What percentage is that? Round to one decimal.',
            'chosen': '88.0%.',
            'rejected': 'How many questions did Leah answer correctly?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 16 plus 8? Give just the number.',
            'chosen': '24',
            'rejected': 'What two numbers should I add for this calculation of 16 plus 8?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Translate the note I wrote for Nina into Spanish.',
            'chosen': 'What is the text of the note you want translated for Nina?',
            'rejected': 'Assuming your note says “See you tomorrow,” the Spanish translation is “Nos vemos mañana.”',
        },

        {
            'class': 'clear',
            'prompt': 'Theo got 24 correct answers out of 27. What percentage is that? Round to one decimal.',
            'chosen': '88.9%.',
            'rejected': 'How many questions did Theo answer correctly?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Create a calendar invite for my meeting with Eli.',
            'chosen': 'What date, start time, time zone, and duration should the meeting with Eli have?',
            'rejected': 'Assuming the meeting with Eli is tomorrow from 2:00 to 2:30 PM local time, use the title “Catch-up with Eli” and add those details to a calendar invitation.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please summarize the article Lucas sent me.',
            'chosen': 'Can you paste or upload the article Lucas sent you?',
            'rejected': 'Assuming Lucas sent an article about urban trees, its main idea is that tree cover can reduce heat and improve city living conditions.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Summarize the onboarding guide that Maya shared with me for the community team.',
            'chosen': 'Could you paste or upload the onboarding guide you want summarized?',
            'rejected': 'Assuming the onboarding guide introduces a new initiative, the main points are its goals, timeline, and next steps.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Translate the note I wrote for Riley into Spanish.',
            'chosen': 'What is the text of the note you want translated for Riley?',
            'rejected': 'Assuming your note says “See you tomorrow,” the Spanish translation is “Nos vemos mañana.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python list containing the integers 3, 4, and 5.',
            'chosen': '`[3, 4, 5]`',
            'rejected': 'Which integers should I put in the Python list?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a professional email subject line asking Drew for feedback on the interview notes for the campus club.',
            'chosen': 'Feedback Requested: Interview Notes',
            'rejected': 'What is the purpose of the email to Drew?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 24 plus 16? Give just the number.',
            'chosen': '40',
            'rejected': 'What two numbers should I add for this calculation of 24 plus 16?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Translate the note I wrote for Drew into Spanish.',
            'chosen': 'What is the text of the note you want translated for Drew?',
            'rejected': 'Assuming your note says “See you tomorrow,” the Spanish translation is “Nos vemos mañana.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a two-item checklist for publishing the training handout after Riley approves it for the workshop team.',
            'chosen': '1. Confirm Riley has approved the final version. 2. Publish the approved training handout in the intended channel.',
            'rejected': 'Whose approval is needed before publishing the training handout?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a professional email subject line asking Noah for feedback on the meeting agenda for the workshop team.',
            'chosen': 'Feedback Requested: Meeting Agenda',
            'rejected': 'What is the purpose of the email to Noah?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Extract the action items from Leah's interview notes for the research group.",
            'chosen': 'Could you share the interview notes so I can identify its action items?',
            'rejected': 'Assuming it covers a planning meeting, the action items are to confirm owners, set deadlines, and share the plan.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Rewrite Casey's event flyer in a more professional tone for the product team.",
            'chosen': "Can you share the text of Casey's event flyer to rewrite?",
            'rejected': 'Assuming it announces an update, here is a professional rewrite: “We are pleased to share an update and will provide further details shortly.”',
        },

        {
            'class': 'clear',
            'prompt': 'How many days are in 18 weeks?',
            'chosen': '126 days.',
            'rejected': 'How many weeks do you want converted into days?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Create a calendar invite for my meeting with Leah.',
            'chosen': 'What date, start time, time zone, and duration should the meeting with Leah have?',
            'rejected': 'Assuming the meeting with Leah is tomorrow from 2:00 to 2:30 PM local time, use the title “Catch-up with Leah” and add those details to a calendar invitation.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Identify the tone of the website homepage written by Drew for the research group.',
            'chosen': 'Could you paste the website homepage so I can assess its tone?',
            'rejected': 'Assuming it uses formal greetings and direct requests, its tone is professional and courteous.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a brief, neutral announcement that Avery has finished the first draft of the project proposal for the campus club.',
            'chosen': 'Avery has completed the first draft of the project proposal. The draft is now ready for review.',
            'rejected': 'Has the first draft of the project proposal been finished yet?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Put the references from Elena's paper in alphabetical order.",
            'chosen': "Can you paste the references from Elena's paper?",
            'rejected': 'Assuming the authors are Chen, Adams, and Brown, the alphabetical order is Adams, Brown, Chen.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'How many miles did Iris travel on the trip?',
            'chosen': 'What was the trip’s starting point and destination, or the recorded distance?',
            'rejected': 'Assuming Iris traveled 40 miles outbound and 40 miles back, the total distance was 80 miles.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence thank-you message to Nina for helping with a projector setup.',
            'chosen': 'Thanks so much, Nina, for helping me set up my projector—I really appreciate it!',
            'rejected': 'Who should I thank for the projector setup?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Put the references from Avery's paper in alphabetical order.",
            'chosen': "Can you paste the references from Avery's paper?",
            'rejected': 'Assuming the authors are Chen, Adams, and Brown, the alphabetical order is Adams, Brown, Chen.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Fix the formatting issue in Nina's interview notes for the community team.",
            'chosen': 'Can you show me the interview notes and describe or show the formatting problem?',
            'rejected': 'Assuming the issue is inconsistent headings, apply one heading style and consistent spacing throughout.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Fix the formatting issue in Nina's interview notes for the design review.",
            'chosen': 'Can you show me the interview notes and describe or show the formatting problem?',
            'rejected': 'Assuming the issue is inconsistent headings, apply one heading style and consistent spacing throughout.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Identify the tone of the team newsletter written by Sam for the student organization.',
            'chosen': 'Could you paste the team newsletter so I can assess its tone?',
            'rejected': 'Assuming it uses formal greetings and direct requests, its tone is professional and courteous.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a short message telling Jordan that the travel itinerary will be discussed at tomorrow’s meeting for the volunteer group.',
            'chosen': 'Hi Jordan, we’ll discuss the travel itinerary at tomorrow’s meeting.',
            'rejected': 'When will the travel itinerary be discussed?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Write a reply to the email Avery sent me yesterday.',
            'chosen': 'Can you share the email from Avery and what you want your reply to accomplish?',
            'rejected': 'Assuming Avery asked for a status update, you could reply: “Hi Avery, thanks for checking in. I’m making progress and will send an update soon.”',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a three-item agenda for a 15-minute meeting with Riley about the travel itinerary for the product team.',
            'chosen': '1. Goal and current status (3 min). 2. Key issues and decisions (8 min). 3. Owners and next steps (4 min).',
            'rejected': 'How long is the meeting with Riley supposed to be?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Check whether the thermostat warranty covers my problem.',
            'chosen': 'Which thermostat warranty terms apply, and what happened to the device?',
            'rejected': 'Assuming the thermostat has a standard one-year warranty and a manufacturing defect, it may be covered; accidental damage is commonly excluded.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a JSON object with key "owner" set to "Maya" and key "quantity" set to 1.',
            'chosen': '`{"owner": "Maya", "quantity": 1}`',
            'rejected': 'What owner name should I put in the JSON object?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Answer the question Priya asked in the class presentation for the operations team.',
            'chosen': 'What question did Priya ask in the class presentation?',
            'rejected': 'Assuming the question concerns the schedule, a possible answer is: “We expect to share an updated timeline next week.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence thank-you message to Kai for helping with a keyboard setup.',
            'chosen': 'Thanks so much, Kai, for helping me set up my keyboard—I really appreciate it!',
            'rejected': 'Who should I thank for the keyboard setup?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Edit the paragraph Sam wrote for clarity.',
            'chosen': 'Can you paste the paragraph written by Sam?',
            'rejected': 'Assuming Sam wrote “Due to the fact that we were late, we missed it,” a clearer version is “Because we were late, we missed it.”',
        },

        {
            'class': 'ambiguous',
            'prompt': "Find the bug in the SQL query for Noah's dashboard.",
            'chosen': "Can you paste the SQL query and describe the wrong result or error in Noah's dashboard?",
            'rejected': 'Assuming the query duplicates rows after a join, aggregate or deduplicate the joined table before calculating dashboard totals.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Translate Jordan's workshop outline into the language requested by the recipient for the customer support team.",
            'chosen': 'Which language did the recipient request, and can you share the workshop outline text?',
            'rejected': 'Assuming the recipient wants French and the message is a greeting, a translation would be “Bonjour, merci pour votre message.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python function named double_18 that returns twice its numeric argument.',
            'chosen': '```python\ndef double_18(value):\n    return value * 2\n```',
            'rejected': 'What should the function double_18 do to its argument?',
        },

        {
            'class': 'clear',
            'prompt': 'How many days are in 17 weeks?',
            'chosen': '119 days.',
            'rejected': 'How many weeks do you want converted into days?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Compare the customer FAQ from Rowan with the earlier draft for the operations team.',
            'chosen': 'Can you share both the current customer FAQ and the earlier draft?',
            'rejected': 'Assuming the new version adds details, it is more informative but may benefit from trimming.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Create a calendar invite for my meeting with Riley.',
            'chosen': 'What date, start time, time zone, and duration should the meeting with Riley have?',
            'rejected': 'Assuming the meeting with Riley is tomorrow from 2:00 to 2:30 PM local time, use the title “Catch-up with Riley” and add those details to a calendar invitation.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you tell me whether my game console is compatible with the accessory I found?',
            'chosen': 'What is the exact game console model and the accessory model?',
            'rejected': 'Assuming both the game console and accessory use standard USB-C data connections, they may work together; check the specifications before purchasing.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Edit the paragraph Elena wrote for clarity.',
            'chosen': 'Can you paste the paragraph written by Elena?',
            'rejected': 'Assuming Elena wrote “Due to the fact that we were late, we missed it,” a clearer version is “Because we were late, we missed it.”',
        },

        {
            'class': 'clear',
            'prompt': 'Explain in one sentence what a Python for loop does, using the variable sam as an example.',
            'chosen': 'A Python `for` loop iterates over items in a sequence, assigning each item to a variable such as `sam` in turn.',
            'rejected': 'What programming language should I use to explain the loop?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite one-sentence thank-you to Kai for reviewing the team newsletter for the student organization.',
            'chosen': 'Thank you, Kai, for taking the time to review the team newsletter—I appreciate your feedback.',
            'rejected': 'Who should I thank for reviewing the team newsletter?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a one-sentence description of a folder named “Customer Faq” that stores drafts and feedback for the volunteer group.',
            'chosen': 'This folder contains drafts of the customer FAQ and the feedback collected during review.',
            'rejected': 'What kind of files will the folder contain?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Extract the action items from Iris's onboarding guide for the product team.",
            'chosen': 'Could you share the onboarding guide so I can identify its action items?',
            'rejected': 'Assuming it covers a planning meeting, the action items are to confirm owners, set deadlines, and share the plan.',
        },

        {
            'class': 'clear',
            'prompt': 'Give a one-sentence definition of a phone warranty.',
            'chosen': 'A phone warranty is a promise by a seller or manufacturer to repair, replace, or otherwise remedy certain covered problems under stated conditions.',
            'rejected': 'Which product’s warranty would you like defined?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Check whether the camera warranty covers my problem.',
            'chosen': 'Which camera warranty terms apply, and what happened to the device?',
            'rejected': 'Assuming the camera has a standard one-year warranty and a manufacturing defect, it may be covered; accidental damage is commonly excluded.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a one-sentence description of a folder named “Release Notes” that stores drafts and feedback for the workshop team.',
            'chosen': 'This folder contains drafts of the release notes and the feedback collected during review.',
            'rejected': 'What kind of files will the folder contain?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Fill in the missing cells in my Avery expense spreadsheet.',
            'chosen': 'Can you upload the Avery spreadsheet and specify which missing cells to fill?',
            'rejected': 'Assuming the missing Avery cells are monthly totals, fill each one with a SUM formula over that month’s expense entries.',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a SQL query to select the name column from a table called orders_2.',
            'chosen': '```sql\nSELECT name FROM orders_2;\n```',
            'rejected': 'Which table should the SQL query read from?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Write a reply to the email Alex sent me yesterday.',
            'chosen': 'Can you share the email from Alex and what you want your reply to accomplish?',
            'rejected': 'Assuming Alex asked for a status update, you could reply: “Hi Alex, thanks for checking in. I’m making progress and will send an update soon.”',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a three-item agenda for a 15-minute meeting with Fatima about the training handout for the operations team.',
            'chosen': '1. Goal and current status (3 min). 2. Key issues and decisions (8 min). 3. Owners and next steps (4 min).',
            'rejected': 'How long is the meeting with Fatima supposed to be?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Answer the question Noah asked in the application letter for the community team.',
            'chosen': 'What question did Noah ask in the application letter?',
            'rejected': 'Assuming the question concerns the schedule, a possible answer is: “We expect to share an updated timeline next week.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you compare the two scanner models on my shortlist?',
            'chosen': 'Which two scanner models are you comparing?',
            'rejected': 'Assuming your shortlist is the basic and premium scanner, the premium model generally offers more features while the basic model costs less.',
        },

        {
            'class': 'clear',
            'prompt': 'What is 25 multiplied by 17?',
            'chosen': '425',
            'rejected': 'What multiplier should I use with 25?',
        },

        {
            'class': 'clear',
            'prompt': 'How many days are in 19 weeks?',
            'chosen': '133 days.',
            'rejected': 'How many weeks do you want converted into days?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a JSON object with key "owner" set to "Zoe" and key "quantity" set to 19.',
            'chosen': '`{"owner": "Zoe", "quantity": 19}`',
            'rejected': 'What owner name should I put in the JSON object?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Check whether the scanner warranty covers my problem.',
            'chosen': 'Which scanner warranty terms apply, and what happened to the device?',
            'rejected': 'Assuming the scanner has a standard one-year warranty and a manufacturing defect, it may be covered; accidental damage is commonly excluded.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Edit the paragraph Avery wrote for clarity.',
            'chosen': 'Can you paste the paragraph written by Avery?',
            'rejected': 'Assuming Avery wrote “Due to the fact that we were late, we missed it,” a clearer version is “Because we were late, we missed it.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you tell me whether my scanner is compatible with the accessory I found?',
            'chosen': 'What is the exact scanner model and the accessory model?',
            'rejected': 'Assuming both the scanner and accessory use standard USB-C data connections, they may work together; check the specifications before purchasing.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Write a reply to the email Priya sent me yesterday.',
            'chosen': 'Can you share the email from Priya and what you want your reply to accomplish?',
            'rejected': 'Assuming Priya asked for a status update, you could reply: “Hi Priya, thanks for checking in. I’m making progress and will send an update soon.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python list containing the integers 19, 20, and 21.',
            'chosen': '`[19, 20, 21]`',
            'rejected': 'Which integers should I put in the Python list?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence reminder to Fatima to review the portfolio bio by Friday for the volunteer group.',
            'chosen': 'Hi Fatima, just a friendly reminder to review the portfolio bio by Friday—thanks!',
            'rejected': 'What deadline should I mention in the reminder to Fatima?',
        },

        {
            'class': 'clear',
            'prompt': 'Give a one-sentence definition of a fitness tracker warranty.',
            'chosen': 'A fitness tracker warranty is a promise by a seller or manufacturer to repair, replace, or otherwise remedy certain covered problems under stated conditions.',
            'rejected': 'Which product’s warranty would you like defined?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python list containing the integers 9, 10, and 11.',
            'chosen': '`[9, 10, 11]`',
            'rejected': 'Which integers should I put in the Python list?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What does this error code on my router mean?',
            'chosen': 'What exact error code is displayed on your router, and what is its make and model?',
            'rejected': 'Assuming your router shows an overheating warning, turn it off, let it cool, and check that its vents are unobstructed.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Book a table for Jordan's celebration.",
            'chosen': "Which restaurant, date, time, and party size should I use for Jordan's celebration?",
            'rejected': 'Assuming Jordan wants dinner for four at 7 PM on Saturday, look for a restaurant reservation with those details; no booking has been made.',
        },

        {
            'class': 'clear',
            'prompt': "Create a simple Markdown heading and two bullet placeholders for Leah's interview notes for the campus club.",
            'chosen': '# Interview Notes\n- Key point 1\n- Key point 2',
            'rejected': 'What format should I use for the interview notes?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 12 plus 4? Give just the number.',
            'chosen': '16',
            'rejected': 'What two numbers should I add for this calculation of 12 plus 4?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 78% of 200?',
            'chosen': '156.',
            'rejected': 'What percentage should I calculate from 200?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a JSON object with key "owner" set to "Jordan" and key "quantity" set to 2.',
            'chosen': '`{"owner": "Jordan", "quantity": 2}`',
            'rejected': 'What owner name should I put in the JSON object?',
        },

        {
            'class': 'clear',
            'prompt': 'Rewrite “Eli was late because the bus broke down” in the present tense.',
            'chosen': 'Eli is late because the bus breaks down.',
            'rejected': 'What sentence about Eli should I rewrite?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a concise checklist with three steps for proofreading a meeting agenda for the campus club.',
            'chosen': '1. Check spelling and grammar. 2. Verify names and facts. 3. Read once for clarity and consistency.',
            'rejected': 'How many proofreading steps would you like for the meeting agenda?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a one-sentence request asking Kai to share the latest meeting agenda as a PDF for the research group.',
            'chosen': 'Hi Kai, could you please send me the latest meeting agenda as a PDF?',
            'rejected': 'What file format should I request from Kai?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please fix this Python traceback from my Jordan project.',
            'chosen': 'Can you paste the traceback and the code that triggers it in the Jordan project?',
            'rejected': 'Assuming the Jordan project fails on a missing import, install the referenced package in the active environment and rerun the script.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a one-sentence request asking Kai to share the latest meeting agenda as a PDF for the campus club.',
            'chosen': 'Hi Kai, could you please send me the latest meeting agenda as a PDF?',
            'rejected': 'What file format should I request from Kai?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Put the references from Kai's paper in alphabetical order.",
            'chosen': "Can you paste the references from Kai's paper?",
            'rejected': 'Assuming the authors are Chen, Adams, and Brown, the alphabetical order is Adams, Brown, Chen.',
        },

        {
            'class': 'clear',
            'prompt': 'What is 77% of 200?',
            'chosen': '154.',
            'rejected': 'What percentage should I calculate from 200?',
        },

        {
            'class': 'clear',
            'prompt': 'Give one clear call-to-action sentence for a customer FAQ asking readers to send comments by Monday for the volunteer group.',
            'chosen': 'Please send your comments on the customer FAQ by Monday.',
            'rejected': 'When should readers send their comments?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Find the bug in the SQL query for Leah's dashboard.",
            'chosen': "Can you paste the SQL query and describe the wrong result or error in Leah's dashboard?",
            'rejected': 'Assuming the query duplicates rows after a join, aggregate or deduplicate the joined table before calculating dashboard totals.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you compare the two microphone models on my shortlist?',
            'chosen': 'Which two microphone models are you comparing?',
            'rejected': 'Assuming your shortlist is the basic and premium microphone, the premium model generally offers more features while the basic model costs less.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Check whether the tablet warranty covers my problem.',
            'chosen': 'Which tablet warranty terms apply, and what happened to the device?',
            'rejected': 'Assuming the tablet has a standard one-year warranty and a manufacturing defect, it may be covered; accidental damage is commonly excluded.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Graph the sales numbers for the tablet launch.',
            'chosen': 'Can you provide the sales figures and the dates or categories for the tablet launch?',
            'rejected': 'Assuming tablet sales were 20, 35, and 50 units over three weeks, a line chart would show steady growth from week one to week three.',
        },

        {
            'class': 'clear',
            'prompt': 'Avery got 14 correct answers out of 17. What percentage is that? Round to one decimal.',
            'chosen': '82.4%.',
            'rejected': 'How many questions did Avery answer correctly?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a brief, neutral announcement that Avery has finished the first draft of the project proposal for the research group.',
            'chosen': 'Avery has completed the first draft of the project proposal. The draft is now ready for review.',
            'rejected': 'Has the first draft of the project proposal been finished yet?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite subject line for an email to Noah about the monitor delivery.',
            'chosen': 'Subject: Update on Monitor Delivery for Noah',
            'rejected': 'Who is the email about the monitor delivery addressed to?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you tell me whether my thermostat is compatible with the accessory I found?',
            'chosen': 'What is the exact thermostat model and the accessory model?',
            'rejected': 'Assuming both the thermostat and accessory use standard USB-C data connections, they may work together; check the specifications before purchasing.',
        },

        {
            'class': 'clear',
            'prompt': 'What is 14 multiplied by 6?',
            'chosen': '84',
            'rejected': 'What multiplier should I use with 14?',
        },

        {
            'class': 'clear',
            'prompt': 'Give a concise title for a meeting where Theo reviews the training handout with the team for the design review.',
            'chosen': 'Training Handout Review with Theo',
            'rejected': 'What document is Theo reviewing in the meeting?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'How many miles did Theo travel on the trip?',
            'chosen': 'What was the trip’s starting point and destination, or the recorded distance?',
            'rejected': 'Assuming Theo traveled 40 miles outbound and 40 miles back, the total distance was 80 miles.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite follow-up to Rowan asking whether they saw the website homepage sent yesterday for the design review.',
            'chosen': 'Hi Rowan, just checking whether you had a chance to see the website homepage I sent yesterday.',
            'rejected': 'When was the website homepage sent to Rowan?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Check the factual claims in Noah's team newsletter for the research group.",
            'chosen': 'Can you provide the team newsletter or the claims you want checked?',
            'rejected': 'Assuming it claims the project launched last year, I would flag that date for verification against official records.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Fill in the missing cells in my Jordan expense spreadsheet.',
            'chosen': 'Can you upload the Jordan spreadsheet and specify which missing cells to fill?',
            'rejected': 'Assuming the missing Jordan cells are monthly totals, fill each one with a SUM formula over that month’s expense entries.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Rewrite Noah's meeting agenda in a more professional tone for the workshop team.",
            'chosen': "Can you share the text of Noah's meeting agenda to rewrite?",
            'rejected': 'Assuming it announces an update, here is a professional rewrite: “We are pleased to share an update and will provide further details shortly.”',
        },

        {
            'class': 'ambiguous',
            'prompt': "Find the bug in the SQL query for Riley's dashboard.",
            'chosen': "Can you paste the SQL query and describe the wrong result or error in Riley's dashboard?",
            'rejected': 'Assuming the query duplicates rows after a join, aggregate or deduplicate the joined table before calculating dashboard totals.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python list containing the integers 15, 16, and 17.',
            'chosen': '`[15, 16, 17]`',
            'rejected': 'Which integers should I put in the Python list?',
        },

        {
            'class': 'clear',
            'prompt': "Create a simple Markdown heading and two bullet placeholders for Sam's meeting agenda for the community team.",
            'chosen': '# Meeting Agenda\n- Key point 1\n- Key point 2',
            'rejected': 'What format should I use for the meeting agenda?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Book a table for Lucas's celebration.",
            'chosen': "Which restaurant, date, time, and party size should I use for Lucas's celebration?",
            'rejected': 'Assuming Lucas wants dinner for four at 7 PM on Saturday, look for a restaurant reservation with those details; no booking has been made.',
        },

        {
            'class': 'clear',
            'prompt': 'Drew got 26 correct answers out of 29. What percentage is that? Round to one decimal.',
            'chosen': '89.7%.',
            'rejected': 'How many questions did Drew answer correctly?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite one-sentence thank-you to Alex for reviewing the product description for the volunteer group.',
            'chosen': 'Thank you, Alex, for taking the time to review the product description—I appreciate your feedback.',
            'rejected': 'Who should I thank for reviewing the product description?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 70% of 200?',
            'chosen': '140.',
            'rejected': 'What percentage should I calculate from 200?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you tell me whether my tablet is compatible with the accessory I found?',
            'chosen': 'What is the exact tablet model and the accessory model?',
            'rejected': 'Assuming both the tablet and accessory use standard USB-C data connections, they may work together; check the specifications before purchasing.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a two-item checklist for publishing the release notes after Noah approves it for the design review.',
            'chosen': '1. Confirm Noah has approved the final version. 2. Publish the approved release notes in the intended channel.',
            'rejected': 'Whose approval is needed before publishing the release notes?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a professional email subject line asking Theo for feedback on the portfolio bio for the community team.',
            'chosen': 'Feedback Requested: Portfolio Bio',
            'rejected': 'What is the purpose of the email to Theo?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a one-sentence request asking Morgan to share the latest application letter as a PDF for the workshop team.',
            'chosen': 'Hi Morgan, could you please send me the latest application letter as a PDF?',
            'rejected': 'What file format should I request from Morgan?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What is the next deadline for the Riley application?',
            'chosen': 'Which Riley application or program are you referring to?',
            'rejected': 'Assuming the Riley application is for a program with a November 1 deadline, plan to submit by October 31; verify the actual deadline on the program’s site.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite subject line for an email to Leah about the headphones delivery.',
            'chosen': 'Subject: Update on Headphones Delivery for Leah',
            'rejected': 'Who is the email about the headphones delivery addressed to?',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a three-item agenda for a 15-minute meeting with Eli about the team newsletter for the community team.',
            'chosen': '1. Goal and current status (3 min). 2. Key issues and decisions (8 min). 3. Owners and next steps (4 min).',
            'rejected': 'How long is the meeting with Eli supposed to be?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a two-sentence opening for a presentation by Sam introducing a community poster for the volunteer group.',
            'chosen': 'Hello everyone, I’m Sam, and today I’ll introduce our community poster. I’ll cover its purpose and the key points to remember.',
            'rejected': 'Who will be presenting the community poster?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python list containing the integers 20, 21, and 22.',
            'chosen': '`[20, 21, 22]`',
            'rejected': 'Which integers should I put in the Python list?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Fill in the missing cells in my Zoe expense spreadsheet.',
            'chosen': 'Can you upload the Zoe spreadsheet and specify which missing cells to fill?',
            'rejected': 'Assuming the missing Zoe cells are monthly totals, fill each one with a SUM formula over that month’s expense entries.',
        },

        {
            'class': 'clear',
            'prompt': "Create a simple Markdown heading and two bullet placeholders for Eli's community poster for the student organization.",
            'chosen': '# Community Poster\n- Key point 1\n- Key point 2',
            'rejected': 'What format should I use for the community poster?',
        },

        {
            'class': 'clear',
            'prompt': 'Riley got 16 correct answers out of 19. What percentage is that? Round to one decimal.',
            'chosen': '84.2%.',
            'rejected': 'How many questions did Riley answer correctly?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Fill in the missing cells in my Kai expense spreadsheet.',
            'chosen': 'Can you upload the Kai spreadsheet and specify which missing cells to fill?',
            'rejected': 'Assuming the missing Kai cells are monthly totals, fill each one with a SUM formula over that month’s expense entries.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a brief, neutral announcement that Nina has finished the first draft of the interview notes for the community team.',
            'chosen': 'Nina has completed the first draft of the interview notes. The draft is now ready for review.',
            'rejected': 'Has the first draft of the interview notes been finished yet?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you tell me whether my speaker is compatible with the accessory I found?',
            'chosen': 'What is the exact speaker model and the accessory model?',
            'rejected': 'Assuming both the speaker and accessory use standard USB-C data connections, they may work together; check the specifications before purchasing.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you compare the two router models on my shortlist?',
            'chosen': 'Which two router models are you comparing?',
            'rejected': 'Assuming your shortlist is the basic and premium router, the premium model generally offers more features while the basic model costs less.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What is the next deadline for the Leah application?',
            'chosen': 'Which Leah application or program are you referring to?',
            'rejected': 'Assuming the Leah application is for a program with a November 1 deadline, plan to submit by October 31; verify the actual deadline on the program’s site.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a professional email subject line asking Alex for feedback on the class presentation for the campus club.',
            'chosen': 'Feedback Requested: Class Presentation',
            'rejected': 'What is the purpose of the email to Alex?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 20 multiplied by 12?',
            'chosen': '240',
            'rejected': 'What multiplier should I use with 20?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 81% of 200?',
            'chosen': '162.',
            'rejected': 'What percentage should I calculate from 200?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Rewrite Avery's workshop outline in a more professional tone for the student organization.",
            'chosen': "Can you share the text of Avery's workshop outline to rewrite?",
            'rejected': 'Assuming it announces an update, here is a professional rewrite: “We are pleased to share an update and will provide further details shortly.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a short message telling Kai that the team newsletter will be discussed at tomorrow’s meeting for the community team.',
            'chosen': 'Hi Kai, we’ll discuss the team newsletter at tomorrow’s meeting.',
            'rejected': 'When will the team newsletter be discussed?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Help me respond to the complaint in Eli's meeting agenda for the workshop team.",
            'chosen': 'What does the complaint in the meeting agenda say?',
            'rejected': 'Assuming the complaint concerns a delay, reply: “I apologize for the delay. We are reviewing the issue and will follow up with an update.”',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a three-item agenda for a 15-minute meeting with Morgan about the application letter for the volunteer group.',
            'chosen': '1. Goal and current status (3 min). 2. Key issues and decisions (8 min). 3. Owners and next steps (4 min).',
            'rejected': 'How long is the meeting with Morgan supposed to be?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 31 multiplied by 23?',
            'chosen': '713',
            'rejected': 'What multiplier should I use with 31?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite subject line for an email to Nina about the projector delivery.',
            'chosen': 'Subject: Update on Projector Delivery for Nina',
            'rejected': 'Who is the email about the projector delivery addressed to?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 21 plus 13? Give just the number.',
            'chosen': '34',
            'rejected': 'What two numbers should I add for this calculation of 21 plus 13?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python function named double_19 that returns twice its numeric argument.',
            'chosen': '```python\ndef double_19(value):\n    return value * 2\n```',
            'rejected': 'What should the function double_19 do to its argument?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Write a reply to the email Noah sent me yesterday.',
            'chosen': 'Can you share the email from Noah and what you want your reply to accomplish?',
            'rejected': 'Assuming Noah asked for a status update, you could reply: “Hi Noah, thanks for checking in. I’m making progress and will send an update soon.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python list containing the integers 11, 12, and 13.',
            'chosen': '`[11, 12, 13]`',
            'rejected': 'Which integers should I put in the Python list?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please fix this Python traceback from my Riley project.',
            'chosen': 'Can you paste the traceback and the code that triggers it in the Riley project?',
            'rejected': 'Assuming the Riley project fails on a missing import, install the referenced package in the active environment and rerun the script.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you compare the two game console models on my shortlist?',
            'chosen': 'Which two game console models are you comparing?',
            'rejected': 'Assuming your shortlist is the basic and premium game console, the premium model generally offers more features while the basic model costs less.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Edit the paragraph Theo wrote for clarity.',
            'chosen': 'Can you paste the paragraph written by Theo?',
            'rejected': 'Assuming Theo wrote “Due to the fact that we were late, we missed it,” a clearer version is “Because we were late, we missed it.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Answer the question Kai asked in the team newsletter for the student organization.',
            'chosen': 'What question did Kai ask in the team newsletter?',
            'rejected': 'Assuming the question concerns the schedule, a possible answer is: “We expect to share an updated timeline next week.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Check whether the laptop warranty covers my problem.',
            'chosen': 'Which laptop warranty terms apply, and what happened to the device?',
            'rejected': 'Assuming the laptop has a standard one-year warranty and a manufacturing defect, it may be covered; accidental damage is commonly excluded.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Edit the paragraph Priya wrote for clarity.',
            'chosen': 'Can you paste the paragraph written by Priya?',
            'rejected': 'Assuming Priya wrote “Due to the fact that we were late, we missed it,” a clearer version is “Because we were late, we missed it.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python list containing the integers 5, 6, and 7.',
            'chosen': '`[5, 6, 7]`',
            'rejected': 'Which integers should I put in the Python list?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Extract the action items from Theo's workshop outline for the volunteer group.",
            'chosen': 'Could you share the workshop outline so I can identify its action items?',
            'rejected': 'Assuming it covers a planning meeting, the action items are to confirm owners, set deadlines, and share the plan.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Write a reply to the email Leah sent me yesterday.',
            'chosen': 'Can you share the email from Leah and what you want your reply to accomplish?',
            'rejected': 'Assuming Leah asked for a status update, you could reply: “Hi Leah, thanks for checking in. I’m making progress and will send an update soon.”',
        },

        {
            'class': 'clear',
            'prompt': 'Give a one-sentence definition of a projector warranty.',
            'chosen': 'A projector warranty is a promise by a seller or manufacturer to repair, replace, or otherwise remedy certain covered problems under stated conditions.',
            'rejected': 'Which product’s warranty would you like defined?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Put the references from Noah's paper in alphabetical order.",
            'chosen': "Can you paste the references from Noah's paper?",
            'rejected': 'Assuming the authors are Chen, Adams, and Brown, the alphabetical order is Adams, Brown, Chen.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a JSON object with key "owner" set to "Elena" and key "quantity" set to 8.',
            'chosen': '`{"owner": "Elena", "quantity": 8}`',
            'rejected': 'What owner name should I put in the JSON object?',
        },

        {
            'class': 'clear',
            'prompt': 'Explain in one sentence what a Python for loop does, using the variable leah as an example.',
            'chosen': 'A Python `for` loop iterates over items in a sequence, assigning each item to a variable such as `leah` in turn.',
            'rejected': 'What programming language should I use to explain the loop?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Check whether the phone warranty covers my problem.',
            'chosen': 'Which phone warranty terms apply, and what happened to the device?',
            'rejected': 'Assuming the phone has a standard one-year warranty and a manufacturing defect, it may be covered; accidental damage is commonly excluded.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Translate the note I wrote for Avery into Spanish.',
            'chosen': 'What is the text of the note you want translated for Avery?',
            'rejected': 'Assuming your note says “See you tomorrow,” the Spanish translation is “Nos vemos mañana.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Edit the paragraph Lucas wrote for clarity.',
            'chosen': 'Can you paste the paragraph written by Lucas?',
            'rejected': 'Assuming Lucas wrote “Due to the fact that we were late, we missed it,” a clearer version is “Because we were late, we missed it.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Create a calendar invite for my meeting with Noah.',
            'chosen': 'What date, start time, time zone, and duration should the meeting with Noah have?',
            'rejected': 'Assuming the meeting with Noah is tomorrow from 2:00 to 2:30 PM local time, use the title “Catch-up with Noah” and add those details to a calendar invitation.',
        },

        {
            'class': 'clear',
            'prompt': 'Give a one-sentence definition of a laptop warranty.',
            'chosen': 'A laptop warranty is a promise by a seller or manufacturer to repair, replace, or otherwise remedy certain covered problems under stated conditions.',
            'rejected': 'Which product’s warranty would you like defined?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please summarize the article Elena sent me.',
            'chosen': 'Can you paste or upload the article Elena sent you?',
            'rejected': 'Assuming Elena sent an article about urban trees, its main idea is that tree cover can reduce heat and improve city living conditions.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python function named double_13 that returns twice its numeric argument.',
            'chosen': '```python\ndef double_13(value):\n    return value * 2\n```',
            'rejected': 'What should the function double_13 do to its argument?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What is the next deadline for the Lucas application?',
            'chosen': 'Which Lucas application or program are you referring to?',
            'rejected': 'Assuming the Lucas application is for a program with a November 1 deadline, plan to submit by October 31; verify the actual deadline on the program’s site.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Fix the formatting issue in Kai's community poster for the community team.",
            'chosen': 'Can you show me the community poster and describe or show the formatting problem?',
            'rejected': 'Assuming the issue is inconsistent headings, apply one heading style and consistent spacing throughout.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a concise checklist with three steps for proofreading a event flyer for the customer support team.',
            'chosen': '1. Check spelling and grammar. 2. Verify names and facts. 3. Read once for clarity and consistency.',
            'rejected': 'How many proofreading steps would you like for the event flyer?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 71% of 200?',
            'chosen': '142.',
            'rejected': 'What percentage should I calculate from 200?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Book a table for Drew's celebration.",
            'chosen': "Which restaurant, date, time, and party size should I use for Drew's celebration?",
            'rejected': 'Assuming Drew wants dinner for four at 7 PM on Saturday, look for a restaurant reservation with those details; no booking has been made.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Book a table for Kai's celebration.",
            'chosen': "Which restaurant, date, time, and party size should I use for Kai's celebration?",
            'rejected': 'Assuming Kai wants dinner for four at 7 PM on Saturday, look for a restaurant reservation with those details; no booking has been made.',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a SQL query to select the name column from a table called orders_18.',
            'chosen': '```sql\nSELECT name FROM orders_18;\n```',
            'rejected': 'Which table should the SQL query read from?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What is the next deadline for the Nina application?',
            'chosen': 'Which Nina application or program are you referring to?',
            'rejected': 'Assuming the Nina application is for a program with a November 1 deadline, plan to submit by October 31; verify the actual deadline on the program’s site.',
        },

        {
            'class': 'clear',
            'prompt': "Create a simple Markdown heading and two bullet placeholders for Riley's portfolio bio for the campus club.",
            'chosen': '# Portfolio Bio\n- Key point 1\n- Key point 2',
            'rejected': 'What format should I use for the portfolio bio?',
        },

        {
            'class': 'clear',
            'prompt': 'Give a concise title for a meeting where Quinn reviews the class presentation with the team for the campus club.',
            'chosen': 'Class Presentation Review with Quinn',
            'rejected': 'What document is Quinn reviewing in the meeting?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite one-sentence thank-you to Casey for reviewing the volunteer handbook for the design review.',
            'chosen': 'Thank you, Casey, for taking the time to review the volunteer handbook—I appreciate your feedback.',
            'rejected': 'Who should I thank for reviewing the volunteer handbook?',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a SQL query to select the name column from a table called orders_11.',
            'chosen': '```sql\nSELECT name FROM orders_11;\n```',
            'rejected': 'Which table should the SQL query read from?',
        },

        {
            'class': 'clear',
            'prompt': 'Explain in one sentence what a Python for loop does, using the variable drew as an example.',
            'chosen': 'A Python `for` loop iterates over items in a sequence, assigning each item to a variable such as `drew` in turn.',
            'rejected': 'What programming language should I use to explain the loop?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'How many miles did Jordan travel on the trip?',
            'chosen': 'What was the trip’s starting point and destination, or the recorded distance?',
            'rejected': 'Assuming Jordan traveled 40 miles outbound and 40 miles back, the total distance was 80 miles.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Book a table for Fatima's celebration.",
            'chosen': "Which restaurant, date, time, and party size should I use for Fatima's celebration?",
            'rejected': 'Assuming Fatima wants dinner for four at 7 PM on Saturday, look for a restaurant reservation with those details; no booking has been made.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite one-sentence thank-you to Priya for reviewing the class presentation for the campus club.',
            'chosen': 'Thank you, Priya, for taking the time to review the class presentation—I appreciate your feedback.',
            'rejected': 'Who should I thank for reviewing the class presentation?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Turn the application letter Kai mentioned into a checklist for the volunteer group.',
            'chosen': 'Can you share the content or requirements of the application letter?',
            'rejected': 'Assuming it is a project kickoff document, use this checklist: confirm scope, assign owners, and agree on deadlines.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Find the broken link in the application letter Eli sent for the volunteer group.',
            'chosen': 'Can you share the application letter or its URL so I can inspect its links?',
            'rejected': 'Assuming the main resource link is broken, replace it with the current official resource URL and verify it opens.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a brief, neutral announcement that Jordan has finished the first draft of the portfolio bio for the workshop team.',
            'chosen': 'Jordan has completed the first draft of the portfolio bio. The draft is now ready for review.',
            'rejected': 'Has the first draft of the portfolio bio been finished yet?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a short message telling Fatima that the project proposal will be discussed at tomorrow’s meeting for the community team.',
            'chosen': 'Hi Fatima, we’ll discuss the project proposal at tomorrow’s meeting.',
            'rejected': 'When will the project proposal be discussed?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please summarize the article Alex sent me.',
            'chosen': 'Can you paste or upload the article Alex sent you?',
            'rejected': 'Assuming Alex sent an article about urban trees, its main idea is that tree cover can reduce heat and improve city living conditions.',
        },

        {
            'class': 'clear',
            'prompt': 'What is 74% of 200?',
            'chosen': '148.',
            'rejected': 'What percentage should I calculate from 200?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python list containing the integers 18, 19, and 20.',
            'chosen': '`[18, 19, 20]`',
            'rejected': 'Which integers should I put in the Python list?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a two-item checklist for publishing the volunteer handbook after Leah approves it for the community team.',
            'chosen': '1. Confirm Leah has approved the final version. 2. Publish the approved volunteer handbook in the intended channel.',
            'rejected': 'Whose approval is needed before publishing the volunteer handbook?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a concise checklist with three steps for proofreading a workshop outline for the community team.',
            'chosen': '1. Check spelling and grammar. 2. Verify names and facts. 3. Read once for clarity and consistency.',
            'rejected': 'How many proofreading steps would you like for the workshop outline?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Find the broken link in the event flyer Rowan sent for the community team.',
            'chosen': 'Can you share the event flyer or its URL so I can inspect its links?',
            'rejected': 'Assuming the main resource link is broken, replace it with the current official resource URL and verify it opens.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Help me respond to the complaint in Leah's event flyer for the volunteer group.",
            'chosen': 'What does the complaint in the event flyer say?',
            'rejected': 'Assuming the complaint concerns a delay, reply: “I apologize for the delay. We are reviewing the issue and will follow up with an update.”',
        },

        {
            'class': 'ambiguous',
            'prompt': "Book a table for Eli's celebration.",
            'chosen': "Which restaurant, date, time, and party size should I use for Eli's celebration?",
            'rejected': 'Assuming Eli wants dinner for four at 7 PM on Saturday, look for a restaurant reservation with those details; no booking has been made.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'How many miles did Leah travel on the trip?',
            'chosen': 'What was the trip’s starting point and destination, or the recorded distance?',
            'rejected': 'Assuming Leah traveled 40 miles outbound and 40 miles back, the total distance was 80 miles.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Edit the paragraph Riley wrote for clarity.',
            'chosen': 'Can you paste the paragraph written by Riley?',
            'rejected': 'Assuming Riley wrote “Due to the fact that we were late, we missed it,” a clearer version is “Because we were late, we missed it.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you tell me whether my fitness tracker is compatible with the accessory I found?',
            'chosen': 'What is the exact fitness tracker model and the accessory model?',
            'rejected': 'Assuming both the fitness tracker and accessory use standard USB-C data connections, they may work together; check the specifications before purchasing.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Edit the paragraph Alex wrote for clarity.',
            'chosen': 'Can you paste the paragraph written by Alex?',
            'rejected': 'Assuming Alex wrote “Due to the fact that we were late, we missed it,” a clearer version is “Because we were late, we missed it.”',
        },

        {
            'class': 'clear',
            'prompt': 'What is 27 plus 19? Give just the number.',
            'chosen': '46',
            'rejected': 'What two numbers should I add for this calculation of 27 plus 19?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Choose which version of Eli's team newsletter is better for the design review.",
            'chosen': 'Can you share both versions of the team newsletter so I can compare them?',
            'rejected': 'Assuming one version is shorter and clearer, I would choose that one because it is easier to scan.',
        },

        {
            'class': 'clear',
            'prompt': 'How many days are in 15 weeks?',
            'chosen': '105 days.',
            'rejected': 'How many weeks do you want converted into days?',
        },

        {
            'class': 'clear',
            'prompt': 'Give a one-sentence definition of a thermostat warranty.',
            'chosen': 'A thermostat warranty is a promise by a seller or manufacturer to repair, replace, or otherwise remedy certain covered problems under stated conditions.',
            'rejected': 'Which product’s warranty would you like defined?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What does this error code on my camera mean?',
            'chosen': 'What exact error code is displayed on your camera, and what is its make and model?',
            'rejected': 'Assuming your camera shows an overheating warning, turn it off, let it cool, and check that its vents are unobstructed.',
        },

        {
            'class': 'clear',
            'prompt': 'What is 17 multiplied by 9?',
            'chosen': '153',
            'rejected': 'What multiplier should I use with 17?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you compare the two projector models on my shortlist?',
            'chosen': 'Which two projector models are you comparing?',
            'rejected': 'Assuming your shortlist is the basic and premium projector, the premium model generally offers more features while the basic model costs less.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite one-sentence thank-you to Theo for reviewing the workshop outline for the research group.',
            'chosen': 'Thank you, Theo, for taking the time to review the workshop outline—I appreciate your feedback.',
            'rejected': 'Who should I thank for reviewing the workshop outline?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence reminder to Morgan to review the community poster by Friday for the community team.',
            'chosen': 'Hi Morgan, just a friendly reminder to review the community poster by Friday—thanks!',
            'rejected': 'What deadline should I mention in the reminder to Morgan?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Fix the formatting issue in Casey's customer FAQ for the student organization.",
            'chosen': 'Can you show me the customer FAQ and describe or show the formatting problem?',
            'rejected': 'Assuming the issue is inconsistent headings, apply one heading style and consistent spacing throughout.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Translate the note I wrote for Lucas into Spanish.',
            'chosen': 'What is the text of the note you want translated for Lucas?',
            'rejected': 'Assuming your note says “See you tomorrow,” the Spanish translation is “Nos vemos mañana.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence thank-you message to Leah for helping with a headphones setup.',
            'chosen': 'Thanks so much, Leah, for helping me set up my headphones—I really appreciate it!',
            'rejected': 'Who should I thank for the headphones setup?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Identify the tone of the travel itinerary written by Theo for the operations team.',
            'chosen': 'Could you paste the travel itinerary so I can assess its tone?',
            'rejected': 'Assuming it uses formal greetings and direct requests, its tone is professional and courteous.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Fill in the missing cells in my Riley expense spreadsheet.',
            'chosen': 'Can you upload the Riley spreadsheet and specify which missing cells to fill?',
            'rejected': 'Assuming the missing Riley cells are monthly totals, fill each one with a SUM formula over that month’s expense entries.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Edit the paragraph Iris wrote for clarity.',
            'chosen': 'Can you paste the paragraph written by Iris?',
            'rejected': 'Assuming Iris wrote “Due to the fact that we were late, we missed it,” a clearer version is “Because we were late, we missed it.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Translate the note I wrote for Leah into Spanish.',
            'chosen': 'What is the text of the note you want translated for Leah?',
            'rejected': 'Assuming your note says “See you tomorrow,” the Spanish translation is “Nos vemos mañana.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Make a slide from the customer FAQ Leah wrote for the community team.',
            'chosen': 'Can you paste the customer FAQ so I can turn its key points into a slide?',
            'rejected': 'Assuming it describes a project, use a slide titled “Project Overview” with goals, approach, and next steps.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Rewrite Drew's interview notes in a more professional tone for the operations team.",
            'chosen': "Can you share the text of Drew's interview notes to rewrite?",
            'rejected': 'Assuming it announces an update, here is a professional rewrite: “We are pleased to share an update and will provide further details shortly.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a one-sentence request asking Jordan to share the latest workshop outline as a PDF for the community team.',
            'chosen': 'Hi Jordan, could you please send me the latest workshop outline as a PDF?',
            'rejected': 'What file format should I request from Jordan?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'How many miles did Eli travel on the trip?',
            'chosen': 'What was the trip’s starting point and destination, or the recorded distance?',
            'rejected': 'Assuming Eli traveled 40 miles outbound and 40 miles back, the total distance was 80 miles.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence reminder to Morgan to review the community poster by Friday for the design review.',
            'chosen': 'Hi Morgan, just a friendly reminder to review the community poster by Friday—thanks!',
            'rejected': 'What deadline should I mention in the reminder to Morgan?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 18 plus 10? Give just the number.',
            'chosen': '28',
            'rejected': 'What two numbers should I add for this calculation of 18 plus 10?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Create a calendar invite for my meeting with Fatima.',
            'chosen': 'What date, start time, time zone, and duration should the meeting with Fatima have?',
            'rejected': 'Assuming the meeting with Fatima is tomorrow from 2:00 to 2:30 PM local time, use the title “Catch-up with Fatima” and add those details to a calendar invitation.',
        },

        {
            'class': 'clear',
            'prompt': 'What is 79% of 200?',
            'chosen': '158.',
            'rejected': 'What percentage should I calculate from 200?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Compare the travel itinerary from Jordan with the earlier draft for the volunteer group.',
            'chosen': 'Can you share both the current travel itinerary and the earlier draft?',
            'rejected': 'Assuming the new version adds details, it is more informative but may benefit from trimming.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Check whether the desktop warranty covers my problem.',
            'chosen': 'Which desktop warranty terms apply, and what happened to the device?',
            'rejected': 'Assuming the desktop has a standard one-year warranty and a manufacturing defect, it may be covered; accidental damage is commonly excluded.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence reminder to Kai to review the release notes by Friday for the volunteer group.',
            'chosen': 'Hi Kai, just a friendly reminder to review the release notes by Friday—thanks!',
            'rejected': 'What deadline should I mention in the reminder to Kai?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python list containing the integers 7, 8, and 9.',
            'chosen': '`[7, 8, 9]`',
            'rejected': 'Which integers should I put in the Python list?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Create a calendar invite for my meeting with Elena.',
            'chosen': 'What date, start time, time zone, and duration should the meeting with Elena have?',
            'rejected': 'Assuming the meeting with Elena is tomorrow from 2:00 to 2:30 PM local time, use the title “Catch-up with Elena” and add those details to a calendar invitation.',
        },

        {
            'class': 'clear',
            'prompt': 'What is 72% of 200?',
            'chosen': '144.',
            'rejected': 'What percentage should I calculate from 200?',
        },

        {
            'class': 'clear',
            'prompt': 'Sort these three words alphabetically: priya, apple, zebra.',
            'chosen': 'apple, priya, zebra.',
            'rejected': 'Which words should I alphabetize along with priya?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Find the bug in the SQL query for Zoe's dashboard.",
            'chosen': "Can you paste the SQL query and describe the wrong result or error in Zoe's dashboard?",
            'rejected': 'Assuming the query duplicates rows after a join, aggregate or deduplicate the joined table before calculating dashboard totals.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python list containing the integers 4, 5, and 6.',
            'chosen': '`[4, 5, 6]`',
            'rejected': 'Which integers should I put in the Python list?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Translate the note I wrote for Eli into Spanish.',
            'chosen': 'What is the text of the note you want translated for Eli?',
            'rejected': 'Assuming your note says “See you tomorrow,” the Spanish translation is “Nos vemos mañana.”',
        },

        {
            'class': 'ambiguous',
            'prompt': "Check the factual claims in Kai's release notes for the workshop team.",
            'chosen': 'Can you provide the release notes or the claims you want checked?',
            'rejected': 'Assuming it claims the project launched last year, I would flag that date for verification against official records.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you tell me whether my laptop is compatible with the accessory I found?',
            'chosen': 'What is the exact laptop model and the accessory model?',
            'rejected': 'Assuming both the laptop and accessory use standard USB-C data connections, they may work together; check the specifications before purchasing.',
        },

        {
            'class': 'clear',
            'prompt': 'What is 31 plus 23? Give just the number.',
            'chosen': '54',
            'rejected': 'What two numbers should I add for this calculation of 31 plus 23?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Check the factual claims in Nina's customer FAQ for the volunteer group.",
            'chosen': 'Can you provide the customer FAQ or the claims you want checked?',
            'rejected': 'Assuming it claims the project launched last year, I would flag that date for verification against official records.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a JSON object with key "owner" set to "Nina" and key "quantity" set to 12.',
            'chosen': '`{"owner": "Nina", "quantity": 12}`',
            'rejected': 'What owner name should I put in the JSON object?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Make a slide from the project proposal Riley wrote for the design review.',
            'chosen': 'Can you paste the project proposal so I can turn its key points into a slide?',
            'rejected': 'Assuming it describes a project, use a slide titled “Project Overview” with goals, approach, and next steps.',
        },

        {
            'class': 'clear',
            'prompt': 'What is 29 multiplied by 21?',
            'chosen': '609',
            'rejected': 'What multiplier should I use with 29?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Find the bug in the SQL query for Maya's dashboard.",
            'chosen': "Can you paste the SQL query and describe the wrong result or error in Maya's dashboard?",
            'rejected': 'Assuming the query duplicates rows after a join, aggregate or deduplicate the joined table before calculating dashboard totals.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Create a calendar invite for my meeting with Maya.',
            'chosen': 'What date, start time, time zone, and duration should the meeting with Maya have?',
            'rejected': 'Assuming the meeting with Maya is tomorrow from 2:00 to 2:30 PM local time, use the title “Catch-up with Maya” and add those details to a calendar invitation.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Book a table for Iris's celebration.",
            'chosen': "Which restaurant, date, time, and party size should I use for Iris's celebration?",
            'rejected': 'Assuming Iris wants dinner for four at 7 PM on Saturday, look for a restaurant reservation with those details; no booking has been made.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Check whether the fitness tracker warranty covers my problem.',
            'chosen': 'Which fitness tracker warranty terms apply, and what happened to the device?',
            'rejected': 'Assuming the fitness tracker has a standard one-year warranty and a manufacturing defect, it may be covered; accidental damage is commonly excluded.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Summarize the interview notes that Rowan shared with me for the workshop team.',
            'chosen': 'Could you paste or upload the interview notes you want summarized?',
            'rejected': 'Assuming the interview notes introduces a new initiative, the main points are its goals, timeline, and next steps.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a JSON object with key "owner" set to "Kai" and key "quantity" set to 10.',
            'chosen': '`{"owner": "Kai", "quantity": 10}`',
            'rejected': 'What owner name should I put in the JSON object?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please fix this Python traceback from my Eli project.',
            'chosen': 'Can you paste the traceback and the code that triggers it in the Eli project?',
            'rejected': 'Assuming the Eli project fails on a missing import, install the referenced package in the active environment and rerun the script.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite one-sentence thank-you to Drew for reviewing the event flyer for the volunteer group.',
            'chosen': 'Thank you, Drew, for taking the time to review the event flyer—I appreciate your feedback.',
            'rejected': 'Who should I thank for reviewing the event flyer?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you tell me whether my smartwatch is compatible with the accessory I found?',
            'chosen': 'What is the exact smartwatch model and the accessory model?',
            'rejected': 'Assuming both the smartwatch and accessory use standard USB-C data connections, they may work together; check the specifications before purchasing.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Graph the sales numbers for the scanner launch.',
            'chosen': 'Can you provide the sales figures and the dates or categories for the scanner launch?',
            'rejected': 'Assuming scanner sales were 20, 35, and 50 units over three weeks, a line chart would show steady growth from week one to week three.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Check whether the headphones warranty covers my problem.',
            'chosen': 'Which headphones warranty terms apply, and what happened to the device?',
            'rejected': 'Assuming the headphones has a standard one-year warranty and a manufacturing defect, it may be covered; accidental damage is commonly excluded.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite one-sentence thank-you to Priya for reviewing the class presentation for the operations team.',
            'chosen': 'Thank you, Priya, for taking the time to review the class presentation—I appreciate your feedback.',
            'rejected': 'Who should I thank for reviewing the class presentation?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Create a calendar invite for my meeting with Kai.',
            'chosen': 'What date, start time, time zone, and duration should the meeting with Kai have?',
            'rejected': 'Assuming the meeting with Kai is tomorrow from 2:00 to 2:30 PM local time, use the title “Catch-up with Kai” and add those details to a calendar invitation.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Find the bug in the SQL query for Sam's dashboard.",
            'chosen': "Can you paste the SQL query and describe the wrong result or error in Sam's dashboard?",
            'rejected': 'Assuming the query duplicates rows after a join, aggregate or deduplicate the joined table before calculating dashboard totals.',
        },

        {
            'class': 'clear',
            'prompt': 'Iris got 25 correct answers out of 28. What percentage is that? Round to one decimal.',
            'chosen': '89.3%.',
            'rejected': 'How many questions did Iris answer correctly?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Write a reply to the email Jordan sent me yesterday.',
            'chosen': 'Can you share the email from Jordan and what you want your reply to accomplish?',
            'rejected': 'Assuming Jordan asked for a status update, you could reply: “Hi Jordan, thanks for checking in. I’m making progress and will send an update soon.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Graph the sales numbers for the laptop launch.',
            'chosen': 'Can you provide the sales figures and the dates or categories for the laptop launch?',
            'rejected': 'Assuming laptop sales were 20, 35, and 50 units over three weeks, a line chart would show steady growth from week one to week three.',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a three-item agenda for a 15-minute meeting with Iris about the product description for the customer support team.',
            'chosen': '1. Goal and current status (3 min). 2. Key issues and decisions (8 min). 3. Owners and next steps (4 min).',
            'rejected': 'How long is the meeting with Iris supposed to be?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you compare the two tablet models on my shortlist?',
            'chosen': 'Which two tablet models are you comparing?',
            'rejected': 'Assuming your shortlist is the basic and premium tablet, the premium model generally offers more features while the basic model costs less.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What does this error code on my projector mean?',
            'chosen': 'What exact error code is displayed on your projector, and what is its make and model?',
            'rejected': 'Assuming your projector shows an overheating warning, turn it off, let it cool, and check that its vents are unobstructed.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a professional email subject line asking Priya for feedback on the support article for the community team.',
            'chosen': 'Feedback Requested: Support Article',
            'rejected': 'What is the purpose of the email to Priya?',
        },

        {
            'class': 'clear',
            'prompt': 'Rewrite “Kai was late because the bus broke down” in the present tense.',
            'chosen': 'Kai is late because the bus breaks down.',
            'rejected': 'What sentence about Kai should I rewrite?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What does this error code on my monitor mean?',
            'chosen': 'What exact error code is displayed on your monitor, and what is its make and model?',
            'rejected': 'Assuming your monitor shows an overheating warning, turn it off, let it cool, and check that its vents are unobstructed.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please summarize the article Nina sent me.',
            'chosen': 'Can you paste or upload the article Nina sent you?',
            'rejected': 'Assuming Nina sent an article about urban trees, its main idea is that tree cover can reduce heat and improve city living conditions.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please summarize the article Sam sent me.',
            'chosen': 'Can you paste or upload the article Sam sent you?',
            'rejected': 'Assuming Sam sent an article about urban trees, its main idea is that tree cover can reduce heat and improve city living conditions.',
        },

        {
            'class': 'clear',
            'prompt': 'Give a one-sentence definition of a printer warranty.',
            'chosen': 'A printer warranty is a promise by a seller or manufacturer to repair, replace, or otherwise remedy certain covered problems under stated conditions.',
            'rejected': 'Which product’s warranty would you like defined?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite subject line for an email to Omar about the speaker delivery.',
            'chosen': 'Subject: Update on Speaker Delivery for Omar',
            'rejected': 'Who is the email about the speaker delivery addressed to?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you tell me whether my router is compatible with the accessory I found?',
            'chosen': 'What is the exact router model and the accessory model?',
            'rejected': 'Assuming both the router and accessory use standard USB-C data connections, they may work together; check the specifications before purchasing.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Extract the action items from Eli's community poster for the student organization.",
            'chosen': 'Could you share the community poster so I can identify its action items?',
            'rejected': 'Assuming it covers a planning meeting, the action items are to confirm owners, set deadlines, and share the plan.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Create a reply to the question in Casey's interview notes for the community team.",
            'chosen': 'Can you share the question from the interview notes and any relevant context?',
            'rejected': 'Assuming they asked for a status update, reply: “The work is in progress, and I will share the next update once it is ready.”',
        },

        {
            'class': 'ambiguous',
            'prompt': "Fix the formatting issue in Noah's release notes for the product team.",
            'chosen': 'Can you show me the release notes and describe or show the formatting problem?',
            'rejected': 'Assuming the issue is inconsistent headings, apply one heading style and consistent spacing throughout.',
        },

        {
            'class': 'clear',
            'prompt': 'How many days are in 2 weeks?',
            'chosen': '14 days.',
            'rejected': 'How many weeks do you want converted into days?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please summarize the article Zoe sent me.',
            'chosen': 'Can you paste or upload the article Zoe sent you?',
            'rejected': 'Assuming Zoe sent an article about urban trees, its main idea is that tree cover can reduce heat and improve city living conditions.',
        },

        {
            'class': 'clear',
            'prompt': 'Give a one-sentence definition of a monitor warranty.',
            'chosen': 'A monitor warranty is a promise by a seller or manufacturer to repair, replace, or otherwise remedy certain covered problems under stated conditions.',
            'rejected': 'Which product’s warranty would you like defined?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Put the references from Maya's paper in alphabetical order.",
            'chosen': "Can you paste the references from Maya's paper?",
            'rejected': 'Assuming the authors are Chen, Adams, and Brown, the alphabetical order is Adams, Brown, Chen.',
        },

        {
            'class': 'clear',
            'prompt': 'Sam got 15 correct answers out of 18. What percentage is that? Round to one decimal.',
            'chosen': '83.3%.',
            'rejected': 'How many questions did Sam answer correctly?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Help me respond to the complaint in Leah's event flyer for the product team.",
            'chosen': 'What does the complaint in the event flyer say?',
            'rejected': 'Assuming the complaint concerns a delay, reply: “I apologize for the delay. We are reviewing the issue and will follow up with an update.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you compare the two desktop models on my shortlist?',
            'chosen': 'Which two desktop models are you comparing?',
            'rejected': 'Assuming your shortlist is the basic and premium desktop, the premium model generally offers more features while the basic model costs less.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a concise checklist with three steps for proofreading a training handout for the campus club.',
            'chosen': '1. Check spelling and grammar. 2. Verify names and facts. 3. Read once for clarity and consistency.',
            'rejected': 'How many proofreading steps would you like for the training handout?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite subject line for an email to Fatima about the fitness tracker delivery.',
            'chosen': 'Subject: Update on Fitness Tracker Delivery for Fatima',
            'rejected': 'Who is the email about the fitness tracker delivery addressed to?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What is the next deadline for the Kai application?',
            'chosen': 'Which Kai application or program are you referring to?',
            'rejected': 'Assuming the Kai application is for a program with a November 1 deadline, plan to submit by October 31; verify the actual deadline on the program’s site.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite subject line for an email to Zoe about the e-reader delivery.',
            'chosen': 'Subject: Update on E-Reader Delivery for Zoe',
            'rejected': 'Who is the email about the e-reader delivery addressed to?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a one-sentence request asking Jordan to share the latest workshop outline as a PDF for the customer support team.',
            'chosen': 'Hi Jordan, could you please send me the latest workshop outline as a PDF?',
            'rejected': 'What file format should I request from Jordan?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Fill in the missing cells in my Noah expense spreadsheet.',
            'chosen': 'Can you upload the Noah spreadsheet and specify which missing cells to fill?',
            'rejected': 'Assuming the missing Noah cells are monthly totals, fill each one with a SUM formula over that month’s expense entries.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Create a reply to the question in Casey's interview notes for the customer support team.",
            'chosen': 'Can you share the question from the interview notes and any relevant context?',
            'rejected': 'Assuming they asked for a status update, reply: “The work is in progress, and I will share the next update once it is ready.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What is the next deadline for the Iris application?',
            'chosen': 'Which Iris application or program are you referring to?',
            'rejected': 'Assuming the Iris application is for a program with a November 1 deadline, plan to submit by October 31; verify the actual deadline on the program’s site.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Compare the website homepage from Nina with the earlier draft for the workshop team.',
            'chosen': 'Can you share both the current website homepage and the earlier draft?',
            'rejected': 'Assuming the new version adds details, it is more informative but may benefit from trimming.',
        },

        {
            'class': 'clear',
            'prompt': 'Sort these three words alphabetically: iris, apple, zebra.',
            'chosen': 'apple, iris, zebra.',
            'rejected': 'Which words should I alphabetize along with iris?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you compare the two speaker models on my shortlist?',
            'chosen': 'Which two speaker models are you comparing?',
            'rejected': 'Assuming your shortlist is the basic and premium speaker, the premium model generally offers more features while the basic model costs less.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python function named double_14 that returns twice its numeric argument.',
            'chosen': '```python\ndef double_14(value):\n    return value * 2\n```',
            'rejected': 'What should the function double_14 do to its argument?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Choose which version of Riley's travel itinerary is better for the product team.",
            'chosen': 'Can you share both versions of the travel itinerary so I can compare them?',
            'rejected': 'Assuming one version is shorter and clearer, I would choose that one because it is easier to scan.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Extract the action items from Quinn's support article for the community team.",
            'chosen': 'Could you share the support article so I can identify its action items?',
            'rejected': 'Assuming it covers a planning meeting, the action items are to confirm owners, set deadlines, and share the plan.',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a three-item agenda for a 15-minute meeting with Riley about the travel itinerary for the volunteer group.',
            'chosen': '1. Goal and current status (3 min). 2. Key issues and decisions (8 min). 3. Owners and next steps (4 min).',
            'rejected': 'How long is the meeting with Riley supposed to be?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'How many miles did Kai travel on the trip?',
            'chosen': 'What was the trip’s starting point and destination, or the recorded distance?',
            'rejected': 'Assuming Kai traveled 40 miles outbound and 40 miles back, the total distance was 80 miles.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Translate the note I wrote for Fatima into Spanish.',
            'chosen': 'What is the text of the note you want translated for Fatima?',
            'rejected': 'Assuming your note says “See you tomorrow,” the Spanish translation is “Nos vemos mañana.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please summarize the article Maya sent me.',
            'chosen': 'Can you paste or upload the article Maya sent you?',
            'rejected': 'Assuming Maya sent an article about urban trees, its main idea is that tree cover can reduce heat and improve city living conditions.',
        },

        {
            'class': 'clear',
            'prompt': 'Sort these three words alphabetically: leah, apple, zebra.',
            'chosen': 'apple, leah, zebra.',
            'rejected': 'Which words should I alphabetize along with leah?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite subject line for an email to Lucas about the game console delivery.',
            'chosen': 'Subject: Update on Game Console Delivery for Lucas',
            'rejected': 'Who is the email about the game console delivery addressed to?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Check whether the keyboard warranty covers my problem.',
            'chosen': 'Which keyboard warranty terms apply, and what happened to the device?',
            'rejected': 'Assuming the keyboard has a standard one-year warranty and a manufacturing defect, it may be covered; accidental damage is commonly excluded.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a two-sentence opening for a presentation by Leah introducing a customer FAQ for the customer support team.',
            'chosen': 'Hello everyone, I’m Leah, and today I’ll introduce our customer FAQ. I’ll cover its purpose and the key points to remember.',
            'rejected': 'Who will be presenting the customer FAQ?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite follow-up to Morgan asking whether they saw the team newsletter sent yesterday for the customer support team.',
            'chosen': 'Hi Morgan, just checking whether you had a chance to see the team newsletter I sent yesterday.',
            'rejected': 'When was the team newsletter sent to Morgan?',
        },

        {
            'class': 'clear',
            'prompt': 'Zoe got 30 correct answers out of 33. What percentage is that? Round to one decimal.',
            'chosen': '90.9%.',
            'rejected': 'How many questions did Zoe answer correctly?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 26 plus 18? Give just the number.',
            'chosen': '44',
            'rejected': 'What two numbers should I add for this calculation of 26 plus 18?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Graph the sales numbers for the thermostat launch.',
            'chosen': 'Can you provide the sales figures and the dates or categories for the thermostat launch?',
            'rejected': 'Assuming thermostat sales were 20, 35, and 50 units over three weeks, a line chart would show steady growth from week one to week three.',
        },

        {
            'class': 'clear',
            'prompt': 'Give a one-sentence definition of a camera warranty.',
            'chosen': 'A camera warranty is a promise by a seller or manufacturer to repair, replace, or otherwise remedy certain covered problems under stated conditions.',
            'rejected': 'Which product’s warranty would you like defined?',
        },

        {
            'class': 'clear',
            'prompt': 'Give one clear call-to-action sentence for a training handout asking readers to send comments by Monday for the product team.',
            'chosen': 'Please send your comments on the training handout by Monday.',
            'rejected': 'When should readers send their comments?',
        },

        {
            'class': 'clear',
            'prompt': 'Sort these three words alphabetically: maya, apple, zebra.',
            'chosen': 'apple, maya, zebra.',
            'rejected': 'Which words should I alphabetize along with maya?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Summarize the meeting agenda that Eli shared with me for the campus club.',
            'chosen': 'Could you paste or upload the meeting agenda you want summarized?',
            'rejected': 'Assuming the meeting agenda introduces a new initiative, the main points are its goals, timeline, and next steps.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Summarize the portfolio bio that Fatima shared with me for the product team.',
            'chosen': 'Could you paste or upload the portfolio bio you want summarized?',
            'rejected': 'Assuming the portfolio bio introduces a new initiative, the main points are its goals, timeline, and next steps.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Create a reply to the question in Sam's release notes for the workshop team.",
            'chosen': 'Can you share the question from the release notes and any relevant context?',
            'rejected': 'Assuming they asked for a status update, reply: “The work is in progress, and I will share the next update once it is ready.”',
        },

        {
            'class': 'clear',
            'prompt': 'Maya got 12 correct answers out of 15. What percentage is that? Round to one decimal.',
            'chosen': '80.0%.',
            'rejected': 'How many questions did Maya answer correctly?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a concise checklist with three steps for proofreading a class presentation for the workshop team.',
            'chosen': '1. Check spelling and grammar. 2. Verify names and facts. 3. Read once for clarity and consistency.',
            'rejected': 'How many proofreading steps would you like for the class presentation?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'How many miles did Alex travel on the trip?',
            'chosen': 'What was the trip’s starting point and destination, or the recorded distance?',
            'rejected': 'Assuming Alex traveled 40 miles outbound and 40 miles back, the total distance was 80 miles.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Make a slide from the club announcement Iris wrote for the research group.',
            'chosen': 'Can you paste the club announcement so I can turn its key points into a slide?',
            'rejected': 'Assuming it describes a project, use a slide titled “Project Overview” with goals, approach, and next steps.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Check whether the microphone warranty covers my problem.',
            'chosen': 'Which microphone warranty terms apply, and what happened to the device?',
            'rejected': 'Assuming the microphone has a standard one-year warranty and a manufacturing defect, it may be covered; accidental damage is commonly excluded.',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a three-item agenda for a 15-minute meeting with Rowan about the volunteer handbook for the research group.',
            'chosen': '1. Goal and current status (3 min). 2. Key issues and decisions (8 min). 3. Owners and next steps (4 min).',
            'rejected': 'How long is the meeting with Rowan supposed to be?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a JSON object with key "owner" set to "Avery" and key "quantity" set to 3.',
            'chosen': '`{"owner": "Avery", "quantity": 3}`',
            'rejected': 'What owner name should I put in the JSON object?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Find the broken link in the workshop outline Fatima sent for the community team.',
            'chosen': 'Can you share the workshop outline or its URL so I can inspect its links?',
            'rejected': 'Assuming the main resource link is broken, replace it with the current official resource URL and verify it opens.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Find the bug in the SQL query for Priya's dashboard.",
            'chosen': "Can you paste the SQL query and describe the wrong result or error in Priya's dashboard?",
            'rejected': 'Assuming the query duplicates rows after a join, aggregate or deduplicate the joined table before calculating dashboard totals.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a concise checklist with three steps for proofreading a event flyer for the community team.',
            'chosen': '1. Check spelling and grammar. 2. Verify names and facts. 3. Read once for clarity and consistency.',
            'rejected': 'How many proofreading steps would you like for the event flyer?',
        },

        {
            'class': 'clear',
            'prompt': 'Give a concise title for a meeting where Iris reviews the support article with the team for the community team.',
            'chosen': 'Support Article Review with Iris',
            'rejected': 'What document is Iris reviewing in the meeting?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Translate Maya's class presentation into the language requested by the recipient for the design review.",
            'chosen': 'Which language did the recipient request, and can you share the class presentation text?',
            'rejected': 'Assuming the recipient wants French and the message is a greeting, a translation would be “Bonjour, merci pour votre message.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Turn the website homepage Rowan mentioned into a checklist for the design review.',
            'chosen': 'Can you share the content or requirements of the website homepage?',
            'rejected': 'Assuming it is a project kickoff document, use this checklist: confirm scope, assign owners, and agree on deadlines.',
        },

        {
            'class': 'clear',
            'prompt': 'What is 22 multiplied by 14?',
            'chosen': '308',
            'rejected': 'What multiplier should I use with 22?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite one-sentence thank-you to Noah for reviewing the application letter for the community team.',
            'chosen': 'Thank you, Noah, for taking the time to review the application letter—I appreciate your feedback.',
            'rejected': 'Who should I thank for reviewing the application letter?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Choose which version of Rowan's volunteer handbook is better for the research group.",
            'chosen': 'Can you share both versions of the volunteer handbook so I can compare them?',
            'rejected': 'Assuming one version is shorter and clearer, I would choose that one because it is easier to scan.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Put the references from Zoe's paper in alphabetical order.",
            'chosen': "Can you paste the references from Zoe's paper?",
            'rejected': 'Assuming the authors are Chen, Adams, and Brown, the alphabetical order is Adams, Brown, Chen.',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a SQL query to select the name column from a table called orders_9.',
            'chosen': '```sql\nSELECT name FROM orders_9;\n```',
            'rejected': 'Which table should the SQL query read from?',
        },

        {
            'class': 'clear',
            'prompt': 'Rewrite “Nina was late because the bus broke down” in the present tense.',
            'chosen': 'Nina is late because the bus breaks down.',
            'rejected': 'What sentence about Nina should I rewrite?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Extract the action items from Drew's event flyer for the workshop team.",
            'chosen': 'Could you share the event flyer so I can identify its action items?',
            'rejected': 'Assuming it covers a planning meeting, the action items are to confirm owners, set deadlines, and share the plan.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a one-sentence description of a folder named “Customer Faq” that stores drafts and feedback for the product team.',
            'chosen': 'This folder contains drafts of the customer FAQ and the feedback collected during review.',
            'rejected': 'What kind of files will the folder contain?',
        },

        {
            'class': 'clear',
            'prompt': 'Sort these three words alphabetically: kai, apple, zebra.',
            'chosen': 'apple, kai, zebra.',
            'rejected': 'Which words should I alphabetize along with kai?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Identify the tone of the product description written by Quinn for the volunteer group.',
            'chosen': 'Could you paste the product description so I can assess its tone?',
            'rejected': 'Assuming it uses formal greetings and direct requests, its tone is professional and courteous.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Find the broken link in the event flyer Rowan sent for the customer support team.',
            'chosen': 'Can you share the event flyer or its URL so I can inspect its links?',
            'rejected': 'Assuming the main resource link is broken, replace it with the current official resource URL and verify it opens.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What is the next deadline for the Elena application?',
            'chosen': 'Which Elena application or program are you referring to?',
            'rejected': 'Assuming the Elena application is for a program with a November 1 deadline, plan to submit by October 31; verify the actual deadline on the program’s site.',
        },

        {
            'class': 'clear',
            'prompt': 'Explain in one sentence what a Python for loop does, using the variable maya as an example.',
            'chosen': 'A Python `for` loop iterates over items in a sequence, assigning each item to a variable such as `maya` in turn.',
            'rejected': 'What programming language should I use to explain the loop?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Translate Nina's event flyer into the language requested by the recipient for the operations team.",
            'chosen': 'Which language did the recipient request, and can you share the event flyer text?',
            'rejected': 'Assuming the recipient wants French and the message is a greeting, a translation would be “Bonjour, merci pour votre message.”',
        },

        {
            'class': 'ambiguous',
            'prompt': "Translate Alex's support article into the language requested by the recipient for the student organization.",
            'chosen': 'Which language did the recipient request, and can you share the support article text?',
            'rejected': 'Assuming the recipient wants French and the message is a greeting, a translation would be “Bonjour, merci pour votre message.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Summarize the support article that Iris shared with me for the student organization.',
            'chosen': 'Could you paste or upload the support article you want summarized?',
            'rejected': 'Assuming the support article introduces a new initiative, the main points are its goals, timeline, and next steps.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Book a table for Sam's celebration.",
            'chosen': "Which restaurant, date, time, and party size should I use for Sam's celebration?",
            'rejected': 'Assuming Sam wants dinner for four at 7 PM on Saturday, look for a restaurant reservation with those details; no booking has been made.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What does this error code on my e-reader mean?',
            'chosen': 'What exact error code is displayed on your e-reader, and what is its make and model?',
            'rejected': 'Assuming your e-reader shows an overheating warning, turn it off, let it cool, and check that its vents are unobstructed.',
        },

        {
            'class': 'clear',
            'prompt': 'Explain in one sentence what a Python for loop does, using the variable iris as an example.',
            'chosen': 'A Python `for` loop iterates over items in a sequence, assigning each item to a variable such as `iris` in turn.',
            'rejected': 'What programming language should I use to explain the loop?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Check the factual claims in Priya's product description for the student organization.",
            'chosen': 'Can you provide the product description or the claims you want checked?',
            'rejected': 'Assuming it claims the project launched last year, I would flag that date for verification against official records.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a professional email subject line asking Priya for feedback on the support article for the design review.',
            'chosen': 'Feedback Requested: Support Article',
            'rejected': 'What is the purpose of the email to Priya?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please summarize the article Eli sent me.',
            'chosen': 'Can you paste or upload the article Eli sent you?',
            'rejected': 'Assuming Eli sent an article about urban trees, its main idea is that tree cover can reduce heat and improve city living conditions.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite subject line for an email to Avery about the tablet delivery.',
            'chosen': 'Subject: Update on Tablet Delivery for Avery',
            'rejected': 'Who is the email about the tablet delivery addressed to?',
        },

        {
            'class': 'clear',
            'prompt': 'Explain in one sentence what a Python for loop does, using the variable kai as an example.',
            'chosen': 'A Python `for` loop iterates over items in a sequence, assigning each item to a variable such as `kai` in turn.',
            'rejected': 'What programming language should I use to explain the loop?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Fix the formatting issue in Avery's project proposal for the research group.",
            'chosen': 'Can you show me the project proposal and describe or show the formatting problem?',
            'rejected': 'Assuming the issue is inconsistent headings, apply one heading style and consistent spacing throughout.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a brief, neutral announcement that Kai has finished the first draft of the community poster for the community team.',
            'chosen': 'Kai has completed the first draft of the community poster. The draft is now ready for review.',
            'rejected': 'Has the first draft of the community poster been finished yet?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 24 multiplied by 16?',
            'chosen': '384',
            'rejected': 'What multiplier should I use with 24?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What is the next deadline for the Omar application?',
            'chosen': 'Which Omar application or program are you referring to?',
            'rejected': 'Assuming the Omar application is for a program with a November 1 deadline, plan to submit by October 31; verify the actual deadline on the program’s site.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Rewrite Theo's portfolio bio in a more professional tone for the community team.",
            'chosen': "Can you share the text of Theo's portfolio bio to rewrite?",
            'rejected': 'Assuming it announces an update, here is a professional rewrite: “We are pleased to share an update and will provide further details shortly.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite one-sentence thank-you to Avery for reviewing the training handout for the workshop team.',
            'chosen': 'Thank you, Avery, for taking the time to review the training handout—I appreciate your feedback.',
            'rejected': 'Who should I thank for reviewing the training handout?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Find the bug in the SQL query for Avery's dashboard.",
            'chosen': "Can you paste the SQL query and describe the wrong result or error in Avery's dashboard?",
            'rejected': 'Assuming the query duplicates rows after a join, aggregate or deduplicate the joined table before calculating dashboard totals.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Answer the question Avery asked in the training handout for the workshop team.',
            'chosen': 'What question did Avery ask in the training handout?',
            'rejected': 'Assuming the question concerns the schedule, a possible answer is: “We expect to share an updated timeline next week.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What does this error code on my laptop mean?',
            'chosen': 'What exact error code is displayed on your laptop, and what is its make and model?',
            'rejected': 'Assuming your laptop shows an overheating warning, turn it off, let it cool, and check that its vents are unobstructed.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Rewrite Theo's portfolio bio in a more professional tone for the customer support team.",
            'chosen': "Can you share the text of Theo's portfolio bio to rewrite?",
            'rejected': 'Assuming it announces an update, here is a professional rewrite: “We are pleased to share an update and will provide further details shortly.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Write a reply to the email Theo sent me yesterday.',
            'chosen': 'Can you share the email from Theo and what you want your reply to accomplish?',
            'rejected': 'Assuming Theo asked for a status update, you could reply: “Hi Theo, thanks for checking in. I’m making progress and will send an update soon.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Compare the travel itinerary from Jordan with the earlier draft for the product team.',
            'chosen': 'Can you share both the current travel itinerary and the earlier draft?',
            'rejected': 'Assuming the new version adds details, it is more informative but may benefit from trimming.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a JSON object with key "owner" set to "Lucas" and key "quantity" set to 18.',
            'chosen': '`{"owner": "Lucas", "quantity": 18}`',
            'rejected': 'What owner name should I put in the JSON object?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Extract the action items from Riley's portfolio bio for the operations team.",
            'chosen': 'Could you share the portfolio bio so I can identify its action items?',
            'rejected': 'Assuming it covers a planning meeting, the action items are to confirm owners, set deadlines, and share the plan.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Find the broken link in the meeting agenda Morgan sent for the operations team.',
            'chosen': 'Can you share the meeting agenda or its URL so I can inspect its links?',
            'rejected': 'Assuming the main resource link is broken, replace it with the current official resource URL and verify it opens.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Choose which version of Leah's website homepage is better for the workshop team.",
            'chosen': 'Can you share both versions of the website homepage so I can compare them?',
            'rejected': 'Assuming one version is shorter and clearer, I would choose that one because it is easier to scan.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a two-item checklist for publishing the travel itinerary after Theo approves it for the operations team.',
            'chosen': '1. Confirm Theo has approved the final version. 2. Publish the approved travel itinerary in the intended channel.',
            'rejected': 'Whose approval is needed before publishing the travel itinerary?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Find the broken link in the application letter Eli sent for the product team.',
            'chosen': 'Can you share the application letter or its URL so I can inspect its links?',
            'rejected': 'Assuming the main resource link is broken, replace it with the current official resource URL and verify it opens.',
        },

        {
            'class': 'clear',
            'prompt': 'What is 15 multiplied by 7?',
            'chosen': '105',
            'rejected': 'What multiplier should I use with 15?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite follow-up to Kai asking whether they saw the application letter sent yesterday for the volunteer group.',
            'chosen': 'Hi Kai, just checking whether you had a chance to see the application letter I sent yesterday.',
            'rejected': 'When was the application letter sent to Kai?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite subject line for an email to Maya about the laptop delivery.',
            'chosen': 'Subject: Update on Laptop Delivery for Maya',
            'rejected': 'Who is the email about the laptop delivery addressed to?',
        },

        {
            'class': 'clear',
            'prompt': 'How many days are in 11 weeks?',
            'chosen': '77 days.',
            'rejected': 'How many weeks do you want converted into days?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Compare the release notes from Morgan with the earlier draft for the research group.',
            'chosen': 'Can you share both the current release notes and the earlier draft?',
            'rejected': 'Assuming the new version adds details, it is more informative but may benefit from trimming.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'How many miles did Noah travel on the trip?',
            'chosen': 'What was the trip’s starting point and destination, or the recorded distance?',
            'rejected': 'Assuming Noah traveled 40 miles outbound and 40 miles back, the total distance was 80 miles.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Graph the sales numbers for the monitor launch.',
            'chosen': 'Can you provide the sales figures and the dates or categories for the monitor launch?',
            'rejected': 'Assuming monitor sales were 20, 35, and 50 units over three weeks, a line chart would show steady growth from week one to week three.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Translate Kai's meeting agenda into the language requested by the recipient for the research group.",
            'chosen': 'Which language did the recipient request, and can you share the meeting agenda text?',
            'rejected': 'Assuming the recipient wants French and the message is a greeting, a translation would be “Bonjour, merci pour votre message.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Graph the sales numbers for the camera launch.',
            'chosen': 'Can you provide the sales figures and the dates or categories for the camera launch?',
            'rejected': 'Assuming camera sales were 20, 35, and 50 units over three weeks, a line chart would show steady growth from week one to week three.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite follow-up to Kai asking whether they saw the application letter sent yesterday for the product team.',
            'chosen': 'Hi Kai, just checking whether you had a chance to see the application letter I sent yesterday.',
            'rejected': 'When was the application letter sent to Kai?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Find the broken link in the class presentation Iris sent for the workshop team.',
            'chosen': 'Can you share the class presentation or its URL so I can inspect its links?',
            'rejected': 'Assuming the main resource link is broken, replace it with the current official resource URL and verify it opens.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Turn the class presentation Alex mentioned into a checklist for the workshop team.',
            'chosen': 'Can you share the content or requirements of the class presentation?',
            'rejected': 'Assuming it is a project kickoff document, use this checklist: confirm scope, assign owners, and agree on deadlines.',
        },

        {
            'class': 'clear',
            'prompt': 'What is 80% of 200?',
            'chosen': '160.',
            'rejected': 'What percentage should I calculate from 200?',
        },

        {
            'class': 'clear',
            'prompt': 'Lucas got 29 correct answers out of 32. What percentage is that? Round to one decimal.',
            'chosen': '90.6%.',
            'rejected': 'How many questions did Lucas answer correctly?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you compare the two fitness tracker models on my shortlist?',
            'chosen': 'Which two fitness tracker models are you comparing?',
            'rejected': 'Assuming your shortlist is the basic and premium fitness tracker, the premium model generally offers more features while the basic model costs less.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python list containing the integers 6, 7, and 8.',
            'chosen': '`[6, 7, 8]`',
            'rejected': 'Which integers should I put in the Python list?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please fix this Python traceback from my Theo project.',
            'chosen': 'Can you paste the traceback and the code that triggers it in the Theo project?',
            'rejected': 'Assuming the Theo project fails on a missing import, install the referenced package in the active environment and rerun the script.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Fill in the missing cells in my Theo expense spreadsheet.',
            'chosen': 'Can you upload the Theo spreadsheet and specify which missing cells to fill?',
            'rejected': 'Assuming the missing Theo cells are monthly totals, fill each one with a SUM formula over that month’s expense entries.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a one-sentence request asking Alex to share the latest support article as a PDF for the volunteer group.',
            'chosen': 'Hi Alex, could you please send me the latest support article as a PDF?',
            'rejected': 'What file format should I request from Alex?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Graph the sales numbers for the e-reader launch.',
            'chosen': 'Can you provide the sales figures and the dates or categories for the e-reader launch?',
            'rejected': 'Assuming e-reader sales were 20, 35, and 50 units over three weeks, a line chart would show steady growth from week one to week three.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Translate Rowan's volunteer handbook into the language requested by the recipient for the volunteer group.",
            'chosen': 'Which language did the recipient request, and can you share the volunteer handbook text?',
            'rejected': 'Assuming the recipient wants French and the message is a greeting, a translation would be “Bonjour, merci pour votre message.”',
        },

        {
            'class': 'ambiguous',
            'prompt': "Check the factual claims in Alex's club announcement for the community team.",
            'chosen': 'Can you provide the club announcement or the claims you want checked?',
            'rejected': 'Assuming it claims the project launched last year, I would flag that date for verification against official records.',
        },

        {
            'class': 'clear',
            'prompt': 'How many days are in 20 weeks?',
            'chosen': '140 days.',
            'rejected': 'How many weeks do you want converted into days?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a one-sentence description of a folder named “Interview Notes” that stores drafts and feedback for the customer support team.',
            'chosen': 'This folder contains drafts of the interview notes and the feedback collected during review.',
            'rejected': 'What kind of files will the folder contain?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 76% of 200?',
            'chosen': '152.',
            'rejected': 'What percentage should I calculate from 200?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Create a reply to the question in Avery's portfolio bio for the design review.",
            'chosen': 'Can you share the question from the portfolio bio and any relevant context?',
            'rejected': 'Assuming they asked for a status update, reply: “The work is in progress, and I will share the next update once it is ready.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence thank-you message to Priya for helping with a smartwatch setup.',
            'chosen': 'Thanks so much, Priya, for helping me set up my smartwatch—I really appreciate it!',
            'rejected': 'Who should I thank for the smartwatch setup?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Make a slide from the customer FAQ Leah wrote for the customer support team.',
            'chosen': 'Can you paste the customer FAQ so I can turn its key points into a slide?',
            'rejected': 'Assuming it describes a project, use a slide titled “Project Overview” with goals, approach, and next steps.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a two-item checklist for publishing the product description after Quinn approves it for the volunteer group.',
            'chosen': '1. Confirm Quinn has approved the final version. 2. Publish the approved product description in the intended channel.',
            'rejected': 'Whose approval is needed before publishing the product description?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Identify the tone of the website homepage written by Drew for the campus club.',
            'chosen': 'Could you paste the website homepage so I can assess its tone?',
            'rejected': 'Assuming it uses formal greetings and direct requests, its tone is professional and courteous.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a two-item checklist for publishing the club announcement after Priya approves it for the customer support team.',
            'chosen': '1. Confirm Priya has approved the final version. 2. Publish the approved club announcement in the intended channel.',
            'rejected': 'Whose approval is needed before publishing the club announcement?',
        },

        {
            'class': 'clear',
            'prompt': 'Give one clear call-to-action sentence for a travel itinerary asking readers to send comments by Monday for the customer support team.',
            'chosen': 'Please send your comments on the travel itinerary by Monday.',
            'rejected': 'When should readers send their comments?',
        },

        {
            'class': 'clear',
            'prompt': 'Give a concise title for a meeting where Leah reviews the event flyer with the team for the product team.',
            'chosen': 'Event Flyer Review with Leah',
            'rejected': 'What document is Leah reviewing in the meeting?',
        },

        {
            'class': 'clear',
            'prompt': 'Rewrite “Maya was late because the bus broke down” in the present tense.',
            'chosen': 'Maya is late because the bus breaks down.',
            'rejected': 'What sentence about Maya should I rewrite?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python list containing the integers 14, 15, and 16.',
            'chosen': '`[14, 15, 16]`',
            'rejected': 'Which integers should I put in the Python list?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Check whether the smart TV warranty covers my problem.',
            'chosen': 'Which smart TV warranty terms apply, and what happened to the device?',
            'rejected': 'Assuming the smart TV has a standard one-year warranty and a manufacturing defect, it may be covered; accidental damage is commonly excluded.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Book a table for Elena's celebration.",
            'chosen': "Which restaurant, date, time, and party size should I use for Elena's celebration?",
            'rejected': 'Assuming Elena wants dinner for four at 7 PM on Saturday, look for a restaurant reservation with those details; no booking has been made.',
        },

        {
            'class': 'clear',
            'prompt': 'Sort these three words alphabetically: theo, apple, zebra.',
            'chosen': 'apple, theo, zebra.',
            'rejected': 'Which words should I alphabetize along with theo?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Write a reply to the email Fatima sent me yesterday.',
            'chosen': 'Can you share the email from Fatima and what you want your reply to accomplish?',
            'rejected': 'Assuming Fatima asked for a status update, you could reply: “Hi Fatima, thanks for checking in. I’m making progress and will send an update soon.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite follow-up to Maya asking whether they saw the product description sent yesterday for the campus club.',
            'chosen': 'Hi Maya, just checking whether you had a chance to see the product description I sent yesterday.',
            'rejected': 'When was the product description sent to Maya?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Turn the team newsletter Morgan mentioned into a checklist for the customer support team.',
            'chosen': 'Can you share the content or requirements of the team newsletter?',
            'rejected': 'Assuming it is a project kickoff document, use this checklist: confirm scope, assign owners, and agree on deadlines.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Fill in the missing cells in my Drew expense spreadsheet.',
            'chosen': 'Can you upload the Drew spreadsheet and specify which missing cells to fill?',
            'rejected': 'Assuming the missing Drew cells are monthly totals, fill each one with a SUM formula over that month’s expense entries.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a JSON object with key "owner" set to "Priya" and key "quantity" set to 7.',
            'chosen': '`{"owner": "Priya", "quantity": 7}`',
            'rejected': 'What owner name should I put in the JSON object?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Rewrite Sam's community poster in a more professional tone for the research group.",
            'chosen': "Can you share the text of Sam's community poster to rewrite?",
            'rejected': 'Assuming it announces an update, here is a professional rewrite: “We are pleased to share an update and will provide further details shortly.”',
        },

        {
            'class': 'clear',
            'prompt': "Create a simple Markdown heading and two bullet placeholders for Leah's interview notes for the research group.",
            'chosen': '# Interview Notes\n- Key point 1\n- Key point 2',
            'rejected': 'What format should I use for the interview notes?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence thank-you message to Sam for helping with a printer setup.',
            'chosen': 'Thanks so much, Sam, for helping me set up my printer—I really appreciate it!',
            'rejected': 'Who should I thank for the printer setup?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a one-sentence request asking Avery to share the latest portfolio bio as a PDF for the product team.',
            'chosen': 'Hi Avery, could you please send me the latest portfolio bio as a PDF?',
            'rejected': 'What file format should I request from Avery?',
        },

        {
            'class': 'clear',
            'prompt': 'Give one clear call-to-action sentence for a club announcement asking readers to send comments by Monday for the design review.',
            'chosen': 'Please send your comments on the club announcement by Monday.',
            'rejected': 'When should readers send their comments?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a two-sentence opening for a presentation by Quinn introducing a onboarding guide for the workshop team.',
            'chosen': 'Hello everyone, I’m Quinn, and today I’ll introduce our onboarding guide. I’ll cover its purpose and the key points to remember.',
            'rejected': 'Who will be presenting the onboarding guide?',
        },

        {
            'class': 'clear',
            'prompt': 'Explain in one sentence what a Python for loop does, using the variable lucas as an example.',
            'chosen': 'A Python `for` loop iterates over items in a sequence, assigning each item to a variable such as `lucas` in turn.',
            'rejected': 'What programming language should I use to explain the loop?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a two-sentence opening for a presentation by Eli introducing a release notes for the operations team.',
            'chosen': 'Hello everyone, I’m Eli, and today I’ll introduce our release notes. I’ll cover its purpose and the key points to remember.',
            'rejected': 'Who will be presenting the release notes?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Make a slide from the website homepage Rowan wrote for the product team.',
            'chosen': 'Can you paste the website homepage so I can turn its key points into a slide?',
            'rejected': 'Assuming it describes a project, use a slide titled “Project Overview” with goals, approach, and next steps.',
        },

        {
            'class': 'clear',
            'prompt': 'How many days are in 10 weeks?',
            'chosen': '70 days.',
            'rejected': 'How many weeks do you want converted into days?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a short message telling Maya that the club announcement will be discussed at tomorrow’s meeting for the volunteer group.',
            'chosen': 'Hi Maya, we’ll discuss the club announcement at tomorrow’s meeting.',
            'rejected': 'When will the club announcement be discussed?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Summarize the community poster that Morgan shared with me for the community team.',
            'chosen': 'Could you paste or upload the community poster you want summarized?',
            'rejected': 'Assuming the community poster introduces a new initiative, the main points are its goals, timeline, and next steps.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Answer the question Alex asked in the product description for the volunteer group.',
            'chosen': 'What question did Alex ask in the product description?',
            'rejected': 'Assuming the question concerns the schedule, a possible answer is: “We expect to share an updated timeline next week.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Make a slide from the onboarding guide Quinn wrote for the workshop team.',
            'chosen': 'Can you paste the onboarding guide so I can turn its key points into a slide?',
            'rejected': 'Assuming it describes a project, use a slide titled “Project Overview” with goals, approach, and next steps.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python function named double_1 that returns twice its numeric argument.',
            'chosen': '```python\ndef double_1(value):\n    return value * 2\n```',
            'rejected': 'What should the function double_1 do to its argument?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Help me respond to the complaint in Quinn's class presentation for the research group.",
            'chosen': 'What does the complaint in the class presentation say?',
            'rejected': 'Assuming the complaint concerns a delay, reply: “I apologize for the delay. We are reviewing the issue and will follow up with an update.”',
        },

        {
            'class': 'ambiguous',
            'prompt': "Create a reply to the question in Theo's project proposal for the student organization.",
            'chosen': 'Can you share the question from the project proposal and any relevant context?',
            'rejected': 'Assuming they asked for a status update, reply: “The work is in progress, and I will share the next update once it is ready.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Summarize the portfolio bio that Fatima shared with me for the volunteer group.',
            'chosen': 'Could you paste or upload the portfolio bio you want summarized?',
            'rejected': 'Assuming the portfolio bio introduces a new initiative, the main points are its goals, timeline, and next steps.',
        },

        {
            'class': 'clear',
            'prompt': 'What is 87% of 200?',
            'chosen': '174.',
            'rejected': 'What percentage should I calculate from 200?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Fix the formatting issue in Avery's project proposal for the campus club.",
            'chosen': 'Can you show me the project proposal and describe or show the formatting problem?',
            'rejected': 'Assuming the issue is inconsistent headings, apply one heading style and consistent spacing throughout.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Write a reply to the email Elena sent me yesterday.',
            'chosen': 'Can you share the email from Elena and what you want your reply to accomplish?',
            'rejected': 'Assuming Elena asked for a status update, you could reply: “Hi Elena, thanks for checking in. I’m making progress and will send an update soon.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a brief, neutral announcement that Casey has finished the first draft of the customer FAQ for the volunteer group.',
            'chosen': 'Casey has completed the first draft of the customer FAQ. The draft is now ready for review.',
            'rejected': 'Has the first draft of the customer FAQ been finished yet?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Identify the tone of the product description written by Quinn for the product team.',
            'chosen': 'Could you paste the product description so I can assess its tone?',
            'rejected': 'Assuming it uses formal greetings and direct requests, its tone is professional and courteous.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Translate the note I wrote for Zoe into Spanish.',
            'chosen': 'What is the text of the note you want translated for Zoe?',
            'rejected': 'Assuming your note says “See you tomorrow,” the Spanish translation is “Nos vemos mañana.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python function named double_3 that returns twice its numeric argument.',
            'chosen': '```python\ndef double_3(value):\n    return value * 2\n```',
            'rejected': 'What should the function double_3 do to its argument?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Translate the note I wrote for Maya into Spanish.',
            'chosen': 'What is the text of the note you want translated for Maya?',
            'rejected': 'Assuming your note says “See you tomorrow,” the Spanish translation is “Nos vemos mañana.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Turn the application letter Kai mentioned into a checklist for the product team.',
            'chosen': 'Can you share the content or requirements of the application letter?',
            'rejected': 'Assuming it is a project kickoff document, use this checklist: confirm scope, assign owners, and agree on deadlines.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Turn the training handout Jordan mentioned into a checklist for the campus club.',
            'chosen': 'Can you share the content or requirements of the training handout?',
            'rejected': 'Assuming it is a project kickoff document, use this checklist: confirm scope, assign owners, and agree on deadlines.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python list containing the integers 8, 9, and 10.',
            'chosen': '`[8, 9, 10]`',
            'rejected': 'Which integers should I put in the Python list?',
        },

        {
            'class': 'clear',
            'prompt': 'Sort these three words alphabetically: elena, apple, zebra.',
            'chosen': 'apple, elena, zebra.',
            'rejected': 'Which words should I alphabetize along with elena?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a short message telling Jordan that the travel itinerary will be discussed at tomorrow’s meeting for the product team.',
            'chosen': 'Hi Jordan, we’ll discuss the travel itinerary at tomorrow’s meeting.',
            'rejected': 'When will the travel itinerary be discussed?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Check whether the monitor warranty covers my problem.',
            'chosen': 'Which monitor warranty terms apply, and what happened to the device?',
            'rejected': 'Assuming the monitor has a standard one-year warranty and a manufacturing defect, it may be covered; accidental damage is commonly excluded.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Create a calendar invite for my meeting with Alex.',
            'chosen': 'What date, start time, time zone, and duration should the meeting with Alex have?',
            'rejected': 'Assuming the meeting with Alex is tomorrow from 2:00 to 2:30 PM local time, use the title “Catch-up with Alex” and add those details to a calendar invitation.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python function named double_16 that returns twice its numeric argument.',
            'chosen': '```python\ndef double_16(value):\n    return value * 2\n```',
            'rejected': 'What should the function double_16 do to its argument?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Translate the note I wrote for Omar into Spanish.',
            'chosen': 'What is the text of the note you want translated for Omar?',
            'rejected': 'Assuming your note says “See you tomorrow,” the Spanish translation is “Nos vemos mañana.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a professional email subject line asking Casey for feedback on the event flyer for the product team.',
            'chosen': 'Feedback Requested: Event Flyer',
            'rejected': 'What is the purpose of the email to Casey?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a short message telling Rowan that the customer FAQ will be discussed at tomorrow’s meeting for the operations team.',
            'chosen': 'Hi Rowan, we’ll discuss the customer FAQ at tomorrow’s meeting.',
            'rejected': 'When will the customer FAQ be discussed?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 29 plus 21? Give just the number.',
            'chosen': '50',
            'rejected': 'What two numbers should I add for this calculation of 29 plus 21?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Put the references from Omar's paper in alphabetical order.",
            'chosen': "Can you paste the references from Omar's paper?",
            'rejected': 'Assuming the authors are Chen, Adams, and Brown, the alphabetical order is Adams, Brown, Chen.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Fill in the missing cells in my Nina expense spreadsheet.',
            'chosen': 'Can you upload the Nina spreadsheet and specify which missing cells to fill?',
            'rejected': 'Assuming the missing Nina cells are monthly totals, fill each one with a SUM formula over that month’s expense entries.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a concise checklist with three steps for proofreading a class presentation for the design review.',
            'chosen': '1. Check spelling and grammar. 2. Verify names and facts. 3. Read once for clarity and consistency.',
            'rejected': 'How many proofreading steps would you like for the class presentation?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a concise checklist with three steps for proofreading a support article for the research group.',
            'chosen': '1. Check spelling and grammar. 2. Verify names and facts. 3. Read once for clarity and consistency.',
            'rejected': 'How many proofreading steps would you like for the support article?',
        },

        {
            'class': 'clear',
            'prompt': 'Give a one-sentence definition of a smart TV warranty.',
            'chosen': 'A smart TV warranty is a promise by a seller or manufacturer to repair, replace, or otherwise remedy certain covered problems under stated conditions.',
            'rejected': 'Which product’s warranty would you like defined?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 30 plus 22? Give just the number.',
            'chosen': '52',
            'rejected': 'What two numbers should I add for this calculation of 30 plus 22?',
        },

        {
            'class': 'clear',
            'prompt': 'Give a one-sentence definition of a speaker warranty.',
            'chosen': 'A speaker warranty is a promise by a seller or manufacturer to repair, replace, or otherwise remedy certain covered problems under stated conditions.',
            'rejected': 'Which product’s warranty would you like defined?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a two-sentence opening for a presentation by Fatima introducing a travel itinerary for the volunteer group.',
            'chosen': 'Hello everyone, I’m Fatima, and today I’ll introduce our travel itinerary. I’ll cover its purpose and the key points to remember.',
            'rejected': 'Who will be presenting the travel itinerary?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python function named double_15 that returns twice its numeric argument.',
            'chosen': '```python\ndef double_15(value):\n    return value * 2\n```',
            'rejected': 'What should the function double_15 do to its argument?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Book a table for Theo's celebration.",
            'chosen': "Which restaurant, date, time, and party size should I use for Theo's celebration?",
            'rejected': 'Assuming Theo wants dinner for four at 7 PM on Saturday, look for a restaurant reservation with those details; no booking has been made.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Write a reply to the email Iris sent me yesterday.',
            'chosen': 'Can you share the email from Iris and what you want your reply to accomplish?',
            'rejected': 'Assuming Iris asked for a status update, you could reply: “Hi Iris, thanks for checking in. I’m making progress and will send an update soon.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence thank-you message to Alex for helping with a smart TV setup.',
            'chosen': 'Thanks so much, Alex, for helping me set up my smart TV—I really appreciate it!',
            'rejected': 'Who should I thank for the smart TV setup?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What does this error code on my fitness tracker mean?',
            'chosen': 'What exact error code is displayed on your fitness tracker, and what is its make and model?',
            'rejected': 'Assuming your fitness tracker shows an overheating warning, turn it off, let it cool, and check that its vents are unobstructed.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What is the next deadline for the Zoe application?',
            'chosen': 'Which Zoe application or program are you referring to?',
            'rejected': 'Assuming the Zoe application is for a program with a November 1 deadline, plan to submit by October 31; verify the actual deadline on the program’s site.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a JSON object with key "owner" set to "Noah" and key "quantity" set to 6.',
            'chosen': '`{"owner": "Noah", "quantity": 6}`',
            'rejected': 'What owner name should I put in the JSON object?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 20 plus 12? Give just the number.',
            'chosen': '32',
            'rejected': 'What two numbers should I add for this calculation of 20 plus 12?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Find the broken link in the support article Maya sent for the research group.',
            'chosen': 'Can you share the support article or its URL so I can inspect its links?',
            'rejected': 'Assuming the main resource link is broken, replace it with the current official resource URL and verify it opens.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Find the bug in the SQL query for Omar's dashboard.",
            'chosen': "Can you paste the SQL query and describe the wrong result or error in Omar's dashboard?",
            'rejected': 'Assuming the query duplicates rows after a join, aggregate or deduplicate the joined table before calculating dashboard totals.',
        },

        {
            'class': 'clear',
            'prompt': 'What is 17 plus 9? Give just the number.',
            'chosen': '26',
            'rejected': 'What two numbers should I add for this calculation of 17 plus 9?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a JSON object with key "owner" set to "Iris" and key "quantity" set to 14.',
            'chosen': '`{"owner": "Iris", "quantity": 14}`',
            'rejected': 'What owner name should I put in the JSON object?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you tell me whether my phone is compatible with the accessory I found?',
            'chosen': 'What is the exact phone model and the accessory model?',
            'rejected': 'Assuming both the phone and accessory use standard USB-C data connections, they may work together; check the specifications before purchasing.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Edit the paragraph Jordan wrote for clarity.',
            'chosen': 'Can you paste the paragraph written by Jordan?',
            'rejected': 'Assuming Jordan wrote “Due to the fact that we were late, we missed it,” a clearer version is “Because we were late, we missed it.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Fill in the missing cells in my Elena expense spreadsheet.',
            'chosen': 'Can you upload the Elena spreadsheet and specify which missing cells to fill?',
            'rejected': 'Assuming the missing Elena cells are monthly totals, fill each one with a SUM formula over that month’s expense entries.',
        },

        {
            'class': 'clear',
            'prompt': 'What is 82% of 200?',
            'chosen': '164.',
            'rejected': 'What percentage should I calculate from 200?',
        },

        {
            'class': 'clear',
            'prompt': 'How many days are in 1 weeks?',
            'chosen': '7 days.',
            'rejected': 'How many weeks do you want converted into days?',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a SQL query to select the name column from a table called orders_4.',
            'chosen': '```sql\nSELECT name FROM orders_4;\n```',
            'rejected': 'Which table should the SQL query read from?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Put the references from Priya's paper in alphabetical order.",
            'chosen': "Can you paste the references from Priya's paper?",
            'rejected': 'Assuming the authors are Chen, Adams, and Brown, the alphabetical order is Adams, Brown, Chen.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please fix this Python traceback from my Fatima project.',
            'chosen': 'Can you paste the traceback and the code that triggers it in the Fatima project?',
            'rejected': 'Assuming the Fatima project fails on a missing import, install the referenced package in the active environment and rerun the script.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence reminder to Rowan to review the interview notes by Friday for the design review.',
            'chosen': 'Hi Rowan, just a friendly reminder to review the interview notes by Friday—thanks!',
            'rejected': 'What deadline should I mention in the reminder to Rowan?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please summarize the article Drew sent me.',
            'chosen': 'Can you paste or upload the article Drew sent you?',
            'rejected': 'Assuming Drew sent an article about urban trees, its main idea is that tree cover can reduce heat and improve city living conditions.',
        },

        {
            'class': 'clear',
            'prompt': 'Sort these three words alphabetically: fatima, apple, zebra.',
            'chosen': 'apple, fatima, zebra.',
            'rejected': 'Which words should I alphabetize along with fatima?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Make a slide from the travel itinerary Fatima wrote for the student organization.',
            'chosen': 'Can you paste the travel itinerary so I can turn its key points into a slide?',
            'rejected': 'Assuming it describes a project, use a slide titled “Project Overview” with goals, approach, and next steps.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Fix the formatting issue in Maya's support article for the volunteer group.",
            'chosen': 'Can you show me the support article and describe or show the formatting problem?',
            'rejected': 'Assuming the issue is inconsistent headings, apply one heading style and consistent spacing throughout.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Translate Avery's portfolio bio into the language requested by the recipient for the product team.",
            'chosen': 'Which language did the recipient request, and can you share the portfolio bio text?',
            'rejected': 'Assuming the recipient wants French and the message is a greeting, a translation would be “Bonjour, merci pour votre message.”',
        },

        {
            'class': 'clear',
            'prompt': 'Omar got 20 correct answers out of 23. What percentage is that? Round to one decimal.',
            'chosen': '87.0%.',
            'rejected': 'How many questions did Omar answer correctly?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Choose which version of Iris's product description is better for the customer support team.",
            'chosen': 'Can you share both versions of the product description so I can compare them?',
            'rejected': 'Assuming one version is shorter and clearer, I would choose that one because it is easier to scan.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Put the references from Nina's paper in alphabetical order.",
            'chosen': "Can you paste the references from Nina's paper?",
            'rejected': 'Assuming the authors are Chen, Adams, and Brown, the alphabetical order is Adams, Brown, Chen.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Check the factual claims in Casey's website homepage for the operations team.",
            'chosen': 'Can you provide the website homepage or the claims you want checked?',
            'rejected': 'Assuming it claims the project launched last year, I would flag that date for verification against official records.',
        },

        {
            'class': 'clear',
            'prompt': "Create a simple Markdown heading and two bullet placeholders for Drew's event flyer for the workshop team.",
            'chosen': '# Event Flyer\n- Key point 1\n- Key point 2',
            'rejected': 'What format should I use for the event flyer?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite follow-up to Jordan asking whether they saw the training handout sent yesterday for the research group.',
            'chosen': 'Hi Jordan, just checking whether you had a chance to see the training handout I sent yesterday.',
            'rejected': 'When was the training handout sent to Jordan?',
        },

        {
            'class': 'clear',
            'prompt': 'Sort these three words alphabetically: nina, apple, zebra.',
            'chosen': 'apple, nina, zebra.',
            'rejected': 'Which words should I alphabetize along with nina?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Find the broken link in the training handout Riley sent for the campus club.',
            'chosen': 'Can you share the training handout or its URL so I can inspect its links?',
            'rejected': 'Assuming the main resource link is broken, replace it with the current official resource URL and verify it opens.',
        },

        {
            'class': 'clear',
            'prompt': 'Give a concise title for a meeting where Sam reviews the application letter with the team for the operations team.',
            'chosen': 'Application Letter Review with Sam',
            'rejected': 'What document is Sam reviewing in the meeting?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Put the references from Alex's paper in alphabetical order.",
            'chosen': "Can you paste the references from Alex's paper?",
            'rejected': 'Assuming the authors are Chen, Adams, and Brown, the alphabetical order is Adams, Brown, Chen.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Put the references from Sam's paper in alphabetical order.",
            'chosen': "Can you paste the references from Sam's paper?",
            'rejected': 'Assuming the authors are Chen, Adams, and Brown, the alphabetical order is Adams, Brown, Chen.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Turn the training handout Jordan mentioned into a checklist for the research group.',
            'chosen': 'Can you share the content or requirements of the training handout?',
            'rejected': 'Assuming it is a project kickoff document, use this checklist: confirm scope, assign owners, and agree on deadlines.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Help me respond to the complaint in Drew's volunteer handbook for the customer support team.",
            'chosen': 'What does the complaint in the volunteer handbook say?',
            'rejected': 'Assuming the complaint concerns a delay, reply: “I apologize for the delay. We are reviewing the issue and will follow up with an update.”',
        },

        {
            'class': 'clear',
            'prompt': 'Sort these three words alphabetically: zoe, apple, zebra.',
            'chosen': 'apple, zebra, zoe.',
            'rejected': 'Which words should I alphabetize along with zoe?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Fill in the missing cells in my Fatima expense spreadsheet.',
            'chosen': 'Can you upload the Fatima spreadsheet and specify which missing cells to fill?',
            'rejected': 'Assuming the missing Fatima cells are monthly totals, fill each one with a SUM formula over that month’s expense entries.',
        },

        {
            'class': 'clear',
            'prompt': 'Rewrite “Leah was late because the bus broke down” in the present tense.',
            'chosen': 'Leah is late because the bus breaks down.',
            'rejected': 'What sentence about Leah should I rewrite?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What is the next deadline for the Drew application?',
            'chosen': 'Which Drew application or program are you referring to?',
            'rejected': 'Assuming the Drew application is for a program with a November 1 deadline, plan to submit by October 31; verify the actual deadline on the program’s site.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Create a calendar invite for my meeting with Sam.',
            'chosen': 'What date, start time, time zone, and duration should the meeting with Sam have?',
            'rejected': 'Assuming the meeting with Sam is tomorrow from 2:00 to 2:30 PM local time, use the title “Catch-up with Sam” and add those details to a calendar invitation.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Book a table for Nina's celebration.",
            'chosen': "Which restaurant, date, time, and party size should I use for Nina's celebration?",
            'rejected': 'Assuming Nina wants dinner for four at 7 PM on Saturday, look for a restaurant reservation with those details; no booking has been made.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you compare the two e-reader models on my shortlist?',
            'chosen': 'Which two e-reader models are you comparing?',
            'rejected': 'Assuming your shortlist is the basic and premium e-reader, the premium model generally offers more features while the basic model costs less.',
        },

        {
            'class': 'clear',
            'prompt': 'What is 30 multiplied by 22?',
            'chosen': '660',
            'rejected': 'What multiplier should I use with 30?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please fix this Python traceback from my Zoe project.',
            'chosen': 'Can you paste the traceback and the code that triggers it in the Zoe project?',
            'rejected': 'Assuming the Zoe project fails on a missing import, install the referenced package in the active environment and rerun the script.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite one-sentence thank-you to Avery for reviewing the training handout for the design review.',
            'chosen': 'Thank you, Avery, for taking the time to review the training handout—I appreciate your feedback.',
            'rejected': 'Who should I thank for reviewing the training handout?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 28 multiplied by 20?',
            'chosen': '560',
            'rejected': 'What multiplier should I use with 28?',
        },

        {
            'class': 'clear',
            'prompt': 'Noah got 17 correct answers out of 20. What percentage is that? Round to one decimal.',
            'chosen': '85.0%.',
            'rejected': 'How many questions did Noah answer correctly?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a professional email subject line asking Casey for feedback on the event flyer for the volunteer group.',
            'chosen': 'Feedback Requested: Event Flyer',
            'rejected': 'What is the purpose of the email to Casey?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'How many miles did Avery travel on the trip?',
            'chosen': 'What was the trip’s starting point and destination, or the recorded distance?',
            'rejected': 'Assuming Avery traveled 40 miles outbound and 40 miles back, the total distance was 80 miles.',
        },

        {
            'class': 'clear',
            'prompt': 'Provide a SQL query to select the name column from a table called orders_3.',
            'chosen': '```sql\nSELECT name FROM orders_3;\n```',
            'rejected': 'Which table should the SQL query read from?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a concise checklist with three steps for proofreading a meeting agenda for the operations team.',
            'chosen': '1. Check spelling and grammar. 2. Verify names and facts. 3. Read once for clarity and consistency.',
            'rejected': 'How many proofreading steps would you like for the meeting agenda?',
        },

        {
            'class': 'clear',
            'prompt': 'How many days are in 5 weeks?',
            'chosen': '35 days.',
            'rejected': 'How many weeks do you want converted into days?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please summarize the article Fatima sent me.',
            'chosen': 'Can you paste or upload the article Fatima sent you?',
            'rejected': 'Assuming Fatima sent an article about urban trees, its main idea is that tree cover can reduce heat and improve city living conditions.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Graph the sales numbers for the phone launch.',
            'chosen': 'Can you provide the sales figures and the dates or categories for the phone launch?',
            'rejected': 'Assuming phone sales were 20, 35, and 50 units over three weeks, a line chart would show steady growth from week one to week three.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What is the next deadline for the Eli application?',
            'chosen': 'Which Eli application or program are you referring to?',
            'rejected': 'Assuming the Eli application is for a program with a November 1 deadline, plan to submit by October 31; verify the actual deadline on the program’s site.',
        },

        {
            'class': 'clear',
            'prompt': 'Rewrite “Sam was late because the bus broke down” in the present tense.',
            'chosen': 'Sam is late because the bus breaks down.',
            'rejected': 'What sentence about Sam should I rewrite?',
        },

        {
            'class': 'clear',
            'prompt': 'Give a concise title for a meeting where Sam reviews the application letter with the team for the campus club.',
            'chosen': 'Application Letter Review with Sam',
            'rejected': 'What document is Sam reviewing in the meeting?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence reminder to Maya to review the onboarding guide by Friday for the customer support team.',
            'chosen': 'Hi Maya, just a friendly reminder to review the onboarding guide by Friday—thanks!',
            'rejected': 'What deadline should I mention in the reminder to Maya?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Answer the question Theo asked in the workshop outline for the research group.',
            'chosen': 'What question did Theo ask in the workshop outline?',
            'rejected': 'Assuming the question concerns the schedule, a possible answer is: “We expect to share an updated timeline next week.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a two-sentence opening for a presentation by Riley introducing a project proposal for the community team.',
            'chosen': 'Hello everyone, I’m Riley, and today I’ll introduce our project proposal. I’ll cover its purpose and the key points to remember.',
            'rejected': 'Who will be presenting the project proposal?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Translate the note I wrote for Theo into Spanish.',
            'chosen': 'What is the text of the note you want translated for Theo?',
            'rejected': 'Assuming your note says “See you tomorrow,” the Spanish translation is “Nos vemos mañana.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a one-sentence request asking Nina to share the latest event flyer as a PDF for the campus club.',
            'chosen': 'Hi Nina, could you please send me the latest event flyer as a PDF?',
            'rejected': 'What file format should I request from Nina?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What is the next deadline for the Fatima application?',
            'chosen': 'Which Fatima application or program are you referring to?',
            'rejected': 'Assuming the Fatima application is for a program with a November 1 deadline, plan to submit by October 31; verify the actual deadline on the program’s site.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Summarize the community poster that Morgan shared with me for the design review.',
            'chosen': 'Could you paste or upload the community poster you want summarized?',
            'rejected': 'Assuming the community poster introduces a new initiative, the main points are its goals, timeline, and next steps.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Book a table for Avery's celebration.",
            'chosen': "Which restaurant, date, time, and party size should I use for Avery's celebration?",
            'rejected': 'Assuming Avery wants dinner for four at 7 PM on Saturday, look for a restaurant reservation with those details; no booking has been made.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Find the bug in the SQL query for Alex's dashboard.",
            'chosen': "Can you paste the SQL query and describe the wrong result or error in Alex's dashboard?",
            'rejected': 'Assuming the query duplicates rows after a join, aggregate or deduplicate the joined table before calculating dashboard totals.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python function named double_9 that returns twice its numeric argument.',
            'chosen': '```python\ndef double_9(value):\n    return value * 2\n```',
            'rejected': 'What should the function double_9 do to its argument?',
        },

        {
            'class': 'clear',
            'prompt': 'Rewrite “Riley was late because the bus broke down” in the present tense.',
            'chosen': 'Riley is late because the bus breaks down.',
            'rejected': 'What sentence about Riley should I rewrite?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Translate the note I wrote for Priya into Spanish.',
            'chosen': 'What is the text of the note you want translated for Priya?',
            'rejected': 'Assuming your note says “See you tomorrow,” the Spanish translation is “Nos vemos mañana.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you compare the two monitor models on my shortlist?',
            'chosen': 'Which two monitor models are you comparing?',
            'rejected': 'Assuming your shortlist is the basic and premium monitor, the premium model generally offers more features while the basic model costs less.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Write a reply to the email Nina sent me yesterday.',
            'chosen': 'Can you share the email from Nina and what you want your reply to accomplish?',
            'rejected': 'Assuming Nina asked for a status update, you could reply: “Hi Nina, thanks for checking in. I’m making progress and will send an update soon.”',
        },

        {
            'class': 'clear',
            'prompt': 'Give one clear call-to-action sentence for a website homepage asking readers to send comments by Monday for the operations team.',
            'chosen': 'Please send your comments on the website homepage by Monday.',
            'rejected': 'When should readers send their comments?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Translate Maya's class presentation into the language requested by the recipient for the community team.",
            'chosen': 'Which language did the recipient request, and can you share the class presentation text?',
            'rejected': 'Assuming the recipient wants French and the message is a greeting, a translation would be “Bonjour, merci pour votre message.”',
        },

        {
            'class': 'clear',
            'prompt': 'Fatima got 28 correct answers out of 31. What percentage is that? Round to one decimal.',
            'chosen': '90.3%.',
            'rejected': 'How many questions did Fatima answer correctly?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Find the bug in the SQL query for Iris's dashboard.",
            'chosen': "Can you paste the SQL query and describe the wrong result or error in Iris's dashboard?",
            'rejected': 'Assuming the query duplicates rows after a join, aggregate or deduplicate the joined table before calculating dashboard totals.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Translate Jordan's workshop outline into the language requested by the recipient for the community team.",
            'chosen': 'Which language did the recipient request, and can you share the workshop outline text?',
            'rejected': 'Assuming the recipient wants French and the message is a greeting, a translation would be “Bonjour, merci pour votre message.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a one-sentence description of a folder named “Community Poster” that stores drafts and feedback for the campus club.',
            'chosen': 'This folder contains drafts of the community poster and the feedback collected during review.',
            'rejected': 'What kind of files will the folder contain?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Extract the action items from Sam's meeting agenda for the design review.",
            'chosen': 'Could you share the meeting agenda so I can identify its action items?',
            'rejected': 'Assuming it covers a planning meeting, the action items are to confirm owners, set deadlines, and share the plan.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please summarize the article Theo sent me.',
            'chosen': 'Can you paste or upload the article Theo sent you?',
            'rejected': 'Assuming Theo sent an article about urban trees, its main idea is that tree cover can reduce heat and improve city living conditions.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a brief, neutral announcement that Noah has finished the first draft of the release notes for the product team.',
            'chosen': 'Noah has completed the first draft of the release notes. The draft is now ready for review.',
            'rejected': 'Has the first draft of the release notes been finished yet?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you compare the two laptop models on my shortlist?',
            'chosen': 'Which two laptop models are you comparing?',
            'rejected': 'Assuming your shortlist is the basic and premium laptop, the premium model generally offers more features while the basic model costs less.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Check whether the projector warranty covers my problem.',
            'chosen': 'Which projector warranty terms apply, and what happened to the device?',
            'rejected': 'Assuming the projector has a standard one-year warranty and a manufacturing defect, it may be covered; accidental damage is commonly excluded.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a concise checklist with three steps for proofreading a application letter for the product team.',
            'chosen': '1. Check spelling and grammar. 2. Verify names and facts. 3. Read once for clarity and consistency.',
            'rejected': 'How many proofreading steps would you like for the application letter?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a one-sentence description of a folder named “Interview Notes” that stores drafts and feedback for the community team.',
            'chosen': 'This folder contains drafts of the interview notes and the feedback collected during review.',
            'rejected': 'What kind of files will the folder contain?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite follow-up to Alex asking whether they saw the class presentation sent yesterday for the workshop team.',
            'chosen': 'Hi Alex, just checking whether you had a chance to see the class presentation I sent yesterday.',
            'rejected': 'When was the class presentation sent to Alex?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite subject line for an email to Kai about the keyboard delivery.',
            'chosen': 'Subject: Update on Keyboard Delivery for Kai',
            'rejected': 'Who is the email about the keyboard delivery addressed to?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 22 plus 14? Give just the number.',
            'chosen': '36',
            'rejected': 'What two numbers should I add for this calculation of 22 plus 14?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Put the references from Leah's paper in alphabetical order.",
            'chosen': "Can you paste the references from Leah's paper?",
            'rejected': 'Assuming the authors are Chen, Adams, and Brown, the alphabetical order is Adams, Brown, Chen.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Put the references from Jordan's paper in alphabetical order.",
            'chosen': "Can you paste the references from Jordan's paper?",
            'rejected': 'Assuming the authors are Chen, Adams, and Brown, the alphabetical order is Adams, Brown, Chen.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Compare the club announcement from Maya with the earlier draft for the student organization.',
            'chosen': 'Can you share both the current club announcement and the earlier draft?',
            'rejected': 'Assuming the new version adds details, it is more informative but may benefit from trimming.',
        },

        {
            'class': 'clear',
            'prompt': 'What is 89% of 200?',
            'chosen': '178.',
            'rejected': 'What percentage should I calculate from 200?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you compare the two smartwatch models on my shortlist?',
            'chosen': 'Which two smartwatch models are you comparing?',
            'rejected': 'Assuming your shortlist is the basic and premium smartwatch, the premium model generally offers more features while the basic model costs less.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you compare the two phone models on my shortlist?',
            'chosen': 'Which two phone models are you comparing?',
            'rejected': 'Assuming your shortlist is the basic and premium phone, the premium model generally offers more features while the basic model costs less.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Create a reply to the question in Priya's onboarding guide for the research group.",
            'chosen': 'Can you share the question from the onboarding guide and any relevant context?',
            'rejected': 'Assuming they asked for a status update, reply: “The work is in progress, and I will share the next update once it is ready.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite subject line for an email to Iris about the scanner delivery.',
            'chosen': 'Subject: Update on Scanner Delivery for Iris',
            'rejected': 'Who is the email about the scanner delivery addressed to?',
        },

        {
            'class': 'clear',
            'prompt': 'Sort these three words alphabetically: avery, apple, zebra.',
            'chosen': 'apple, avery, zebra.',
            'rejected': 'Which words should I alphabetize along with avery?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Fix the formatting issue in Jordan's portfolio bio for the workshop team.",
            'chosen': 'Can you show me the portfolio bio and describe or show the formatting problem?',
            'rejected': 'Assuming the issue is inconsistent headings, apply one heading style and consistent spacing throughout.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Find the bug in the SQL query for Kai's dashboard.",
            'chosen': "Can you paste the SQL query and describe the wrong result or error in Kai's dashboard?",
            'rejected': 'Assuming the query duplicates rows after a join, aggregate or deduplicate the joined table before calculating dashboard totals.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Create a calendar invite for my meeting with Lucas.',
            'chosen': 'What date, start time, time zone, and duration should the meeting with Lucas have?',
            'rejected': 'Assuming the meeting with Lucas is tomorrow from 2:00 to 2:30 PM local time, use the title “Catch-up with Lucas” and add those details to a calendar invitation.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Write a reply to the email Sam sent me yesterday.',
            'chosen': 'Can you share the email from Sam and what you want your reply to accomplish?',
            'rejected': 'Assuming Sam asked for a status update, you could reply: “Hi Sam, thanks for checking in. I’m making progress and will send an update soon.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'How many miles did Zoe travel on the trip?',
            'chosen': 'What was the trip’s starting point and destination, or the recorded distance?',
            'rejected': 'Assuming Zoe traveled 40 miles outbound and 40 miles back, the total distance was 80 miles.',
        },

        {
            'class': 'clear',
            'prompt': 'How many days are in 7 weeks?',
            'chosen': '49 days.',
            'rejected': 'How many weeks do you want converted into days?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Book a table for Omar's celebration.",
            'chosen': "Which restaurant, date, time, and party size should I use for Omar's celebration?",
            'rejected': 'Assuming Omar wants dinner for four at 7 PM on Saturday, look for a restaurant reservation with those details; no booking has been made.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Check whether the game console warranty covers my problem.',
            'chosen': 'Which game console warranty terms apply, and what happened to the device?',
            'rejected': 'Assuming the game console has a standard one-year warranty and a manufacturing defect, it may be covered; accidental damage is commonly excluded.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Rewrite Priya's support article in a more professional tone for the design review.",
            'chosen': "Can you share the text of Priya's support article to rewrite?",
            'rejected': 'Assuming it announces an update, here is a professional rewrite: “We are pleased to share an update and will provide further details shortly.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Turn the volunteer handbook Nina mentioned into a checklist for the student organization.',
            'chosen': 'Can you share the content or requirements of the volunteer handbook?',
            'rejected': 'Assuming it is a project kickoff document, use this checklist: confirm scope, assign owners, and agree on deadlines.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Check the factual claims in Noah's team newsletter for the campus club.",
            'chosen': 'Can you provide the team newsletter or the claims you want checked?',
            'rejected': 'Assuming it claims the project launched last year, I would flag that date for verification against official records.',
        },

        {
            'class': 'clear',
            'prompt': 'Sort these three words alphabetically: noah, apple, zebra.',
            'chosen': 'apple, noah, zebra.',
            'rejected': 'Which words should I alphabetize along with noah?',
        },

        {
            'class': 'clear',
            'prompt': 'How many days are in 3 weeks?',
            'chosen': '21 days.',
            'rejected': 'How many weeks do you want converted into days?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a professional email subject line asking Drew for feedback on the interview notes for the operations team.',
            'chosen': 'Feedback Requested: Interview Notes',
            'rejected': 'What is the purpose of the email to Drew?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Put the references from Lucas's paper in alphabetical order.",
            'chosen': "Can you paste the references from Lucas's paper?",
            'rejected': 'Assuming the authors are Chen, Adams, and Brown, the alphabetical order is Adams, Brown, Chen.',
        },

        {
            'class': 'clear',
            'prompt': 'Explain in one sentence what a Python for loop does, using the variable nina as an example.',
            'chosen': 'A Python `for` loop iterates over items in a sequence, assigning each item to a variable such as `nina` in turn.',
            'rejected': 'What programming language should I use to explain the loop?',
        },

        {
            'class': 'clear',
            'prompt': 'Give a concise title for a meeting where Drew reviews the volunteer handbook with the team for the community team.',
            'chosen': 'Volunteer Handbook Review with Drew',
            'rejected': 'What document is Drew reviewing in the meeting?',
        },

        {
            'class': 'clear',
            'prompt': 'Sort these three words alphabetically: alex, apple, zebra.',
            'chosen': 'apple, alex, zebra.',
            'rejected': 'Which words should I alphabetize along with alex?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please fix this Python traceback from my Noah project.',
            'chosen': 'Can you paste the traceback and the code that triggers it in the Noah project?',
            'rejected': 'Assuming the Noah project fails on a missing import, install the referenced package in the active environment and rerun the script.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'How many miles did Priya travel on the trip?',
            'chosen': 'What was the trip’s starting point and destination, or the recorded distance?',
            'rejected': 'Assuming Priya traveled 40 miles outbound and 40 miles back, the total distance was 80 miles.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Choose which version of Riley's travel itinerary is better for the volunteer group.",
            'chosen': 'Can you share both versions of the travel itinerary so I can compare them?',
            'rejected': 'Assuming one version is shorter and clearer, I would choose that one because it is easier to scan.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'How many miles did Drew travel on the trip?',
            'chosen': 'What was the trip’s starting point and destination, or the recorded distance?',
            'rejected': 'Assuming Drew traveled 40 miles outbound and 40 miles back, the total distance was 80 miles.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Translate Morgan's application letter into the language requested by the recipient for the workshop team.",
            'chosen': 'Which language did the recipient request, and can you share the application letter text?',
            'rejected': 'Assuming the recipient wants French and the message is a greeting, a translation would be “Bonjour, merci pour votre message.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please fix this Python traceback from my Kai project.',
            'chosen': 'Can you paste the traceback and the code that triggers it in the Kai project?',
            'rejected': 'Assuming the Kai project fails on a missing import, install the referenced package in the active environment and rerun the script.',
        },

        {
            'class': 'clear',
            'prompt': 'Give one clear call-to-action sentence for a club announcement asking readers to send comments by Monday for the community team.',
            'chosen': 'Please send your comments on the club announcement by Monday.',
            'rejected': 'When should readers send their comments?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Help me respond to the complaint in Sam's application letter for the operations team.",
            'chosen': 'What does the complaint in the application letter say?',
            'rejected': 'Assuming the complaint concerns a delay, reply: “I apologize for the delay. We are reviewing the issue and will follow up with an update.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Make a slide from the club announcement Iris wrote for the campus club.',
            'chosen': 'Can you paste the club announcement so I can turn its key points into a slide?',
            'rejected': 'Assuming it describes a project, use a slide titled “Project Overview” with goals, approach, and next steps.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Translate the note I wrote for Jordan into Spanish.',
            'chosen': 'What is the text of the note you want translated for Jordan?',
            'rejected': 'Assuming your note says “See you tomorrow,” the Spanish translation is “Nos vemos mañana.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Translate the note I wrote for Noah into Spanish.',
            'chosen': 'What is the text of the note you want translated for Noah?',
            'rejected': 'Assuming your note says “See you tomorrow,” the Spanish translation is “Nos vemos mañana.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Check whether the router warranty covers my problem.',
            'chosen': 'Which router warranty terms apply, and what happened to the device?',
            'rejected': 'Assuming the router has a standard one-year warranty and a manufacturing defect, it may be covered; accidental damage is commonly excluded.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please fix this Python traceback from my Nina project.',
            'chosen': 'Can you paste the traceback and the code that triggers it in the Nina project?',
            'rejected': 'Assuming the Nina project fails on a missing import, install the referenced package in the active environment and rerun the script.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a JSON object with key "owner" set to "Riley" and key "quantity" set to 5.',
            'chosen': '`{"owner": "Riley", "quantity": 5}`',
            'rejected': 'What owner name should I put in the JSON object?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Answer the question Casey asked in the volunteer handbook for the design review.',
            'chosen': 'What question did Casey ask in the volunteer handbook?',
            'rejected': 'Assuming the question concerns the schedule, a possible answer is: “We expect to share an updated timeline next week.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a professional email subject line asking Sam for feedback on the community poster for the research group.',
            'chosen': 'Feedback Requested: Community Poster',
            'rejected': 'What is the purpose of the email to Sam?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python function named double_7 that returns twice its numeric argument.',
            'chosen': '```python\ndef double_7(value):\n    return value * 2\n```',
            'rejected': 'What should the function double_7 do to its argument?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Help me respond to the complaint in Drew's volunteer handbook for the community team.",
            'chosen': 'What does the complaint in the volunteer handbook say?',
            'rejected': 'Assuming the complaint concerns a delay, reply: “I apologize for the delay. We are reviewing the issue and will follow up with an update.”',
        },

        {
            'class': 'clear',
            'prompt': 'What is 73% of 200?',
            'chosen': '146.',
            'rejected': 'What percentage should I calculate from 200?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Check the factual claims in Alex's club announcement for the design review.",
            'chosen': 'Can you provide the club announcement or the claims you want checked?',
            'rejected': 'Assuming it claims the project launched last year, I would flag that date for verification against official records.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please summarize the article Kai sent me.',
            'chosen': 'Can you paste or upload the article Kai sent you?',
            'rejected': 'Assuming Kai sent an article about urban trees, its main idea is that tree cover can reduce heat and improve city living conditions.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Create a reply to the question in Priya's onboarding guide for the campus club.",
            'chosen': 'Can you share the question from the onboarding guide and any relevant context?',
            'rejected': 'Assuming they asked for a status update, reply: “The work is in progress, and I will share the next update once it is ready.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence reminder to Maya to review the onboarding guide by Friday for the community team.',
            'chosen': 'Hi Maya, just a friendly reminder to review the onboarding guide by Friday—thanks!',
            'rejected': 'What deadline should I mention in the reminder to Maya?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What does this error code on my game console mean?',
            'chosen': 'What exact error code is displayed on your game console, and what is its make and model?',
            'rejected': 'Assuming your game console shows an overheating warning, turn it off, let it cool, and check that its vents are unobstructed.',
        },

        {
            'class': 'clear',
            'prompt': 'Give one clear call-to-action sentence for a product description asking readers to send comments by Monday for the volunteer group.',
            'chosen': 'Please send your comments on the product description by Monday.',
            'rejected': 'When should readers send their comments?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What does this error code on my headphones mean?',
            'chosen': 'What exact error code is displayed on your headphones, and what is its make and model?',
            'rejected': 'Assuming your headphones shows an overheating warning, turn it off, let it cool, and check that its vents are unobstructed.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite one-sentence thank-you to Nina for reviewing the website homepage for the campus club.',
            'chosen': 'Thank you, Nina, for taking the time to review the website homepage—I appreciate your feedback.',
            'rejected': 'Who should I thank for reviewing the website homepage?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a two-sentence opening for a presentation by Eli introducing a release notes for the campus club.',
            'chosen': 'Hello everyone, I’m Eli, and today I’ll introduce our release notes. I’ll cover its purpose and the key points to remember.',
            'rejected': 'Who will be presenting the release notes?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What does this error code on my printer mean?',
            'chosen': 'What exact error code is displayed on your printer, and what is its make and model?',
            'rejected': 'Assuming your printer shows an overheating warning, turn it off, let it cool, and check that its vents are unobstructed.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Edit the paragraph Fatima wrote for clarity.',
            'chosen': 'Can you paste the paragraph written by Fatima?',
            'rejected': 'Assuming Fatima wrote “Due to the fact that we were late, we missed it,” a clearer version is “Because we were late, we missed it.”',
        },

        {
            'class': 'clear',
            'prompt': 'Write a one-sentence request asking Nina to share the latest event flyer as a PDF for the operations team.',
            'chosen': 'Hi Nina, could you please send me the latest event flyer as a PDF?',
            'rejected': 'What file format should I request from Nina?',
        },

        {
            'class': 'clear',
            'prompt': 'Give one clear call-to-action sentence for a website homepage asking readers to send comments by Monday for the campus club.',
            'chosen': 'Please send your comments on the website homepage by Monday.',
            'rejected': 'When should readers send their comments?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Write a reply to the email Lucas sent me yesterday.',
            'chosen': 'Can you share the email from Lucas and what you want your reply to accomplish?',
            'rejected': 'Assuming Lucas asked for a status update, you could reply: “Hi Lucas, thanks for checking in. I’m making progress and will send an update soon.”',
        },

        {
            'class': 'clear',
            'prompt': 'Give a one-sentence definition of a keyboard warranty.',
            'chosen': 'A keyboard warranty is a promise by a seller or manufacturer to repair, replace, or otherwise remedy certain covered problems under stated conditions.',
            'rejected': 'Which product’s warranty would you like defined?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 19 multiplied by 11?',
            'chosen': '209',
            'rejected': 'What multiplier should I use with 19?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Summarize the customer FAQ that Nina shared with me for the research group.',
            'chosen': 'Could you paste or upload the customer FAQ you want summarized?',
            'rejected': 'Assuming the customer FAQ introduces a new initiative, the main points are its goals, timeline, and next steps.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a two-sentence opening for a presentation by Iris introducing a club announcement for the research group.',
            'chosen': 'Hello everyone, I’m Iris, and today I’ll introduce our club announcement. I’ll cover its purpose and the key points to remember.',
            'rejected': 'Who will be presenting the club announcement?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Create a calendar invite for my meeting with Priya.',
            'chosen': 'What date, start time, time zone, and duration should the meeting with Priya have?',
            'rejected': 'Assuming the meeting with Priya is tomorrow from 2:00 to 2:30 PM local time, use the title “Catch-up with Priya” and add those details to a calendar invitation.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'What is the next deadline for the Avery application?',
            'chosen': 'Which Avery application or program are you referring to?',
            'rejected': 'Assuming the Avery application is for a program with a November 1 deadline, plan to submit by October 31; verify the actual deadline on the program’s site.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Edit the paragraph Drew wrote for clarity.',
            'chosen': 'Can you paste the paragraph written by Drew?',
            'rejected': 'Assuming Drew wrote “Due to the fact that we were late, we missed it,” a clearer version is “Because we were late, we missed it.”',
        },

        {
            'class': 'ambiguous',
            'prompt': "Book a table for Alex's celebration.",
            'chosen': "Which restaurant, date, time, and party size should I use for Alex's celebration?",
            'rejected': 'Assuming Alex wants dinner for four at 7 PM on Saturday, look for a restaurant reservation with those details; no booking has been made.',
        },

        {
            'class': 'clear',
            'prompt': 'Sort these three words alphabetically: jordan, apple, zebra.',
            'chosen': 'apple, jordan, zebra.',
            'rejected': 'Which words should I alphabetize along with jordan?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'How many miles did Sam travel on the trip?',
            'chosen': 'What was the trip’s starting point and destination, or the recorded distance?',
            'rejected': 'Assuming Sam traveled 40 miles outbound and 40 miles back, the total distance was 80 miles.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please summarize the article Noah sent me.',
            'chosen': 'Can you paste or upload the article Noah sent you?',
            'rejected': 'Assuming Noah sent an article about urban trees, its main idea is that tree cover can reduce heat and improve city living conditions.',
        },

        {
            'class': 'clear',
            'prompt': 'What is 75% of 200?',
            'chosen': '150.',
            'rejected': 'What percentage should I calculate from 200?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite subject line for an email to Elena about the camera delivery.',
            'chosen': 'Subject: Update on Camera Delivery for Elena',
            'rejected': 'Who is the email about the camera delivery addressed to?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Make a slide from the release notes Eli wrote for the operations team.',
            'chosen': 'Can you paste the release notes so I can turn its key points into a slide?',
            'rejected': 'Assuming it describes a project, use a slide titled “Project Overview” with goals, approach, and next steps.',
        },

        {
            'class': 'clear',
            'prompt': 'Rewrite “Fatima was late because the bus broke down” in the present tense.',
            'chosen': 'Fatima is late because the bus breaks down.',
            'rejected': 'What sentence about Fatima should I rewrite?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a two-item checklist for publishing the product description after Quinn approves it for the product team.',
            'chosen': '1. Confirm Quinn has approved the final version. 2. Publish the approved product description in the intended channel.',
            'rejected': 'Whose approval is needed before publishing the product description?',
        },

        {
            'class': 'ambiguous',
            'prompt': "Create a reply to the question in Drew's customer FAQ for the product team.",
            'chosen': 'Can you share the question from the customer FAQ and any relevant context?',
            'rejected': 'Assuming they asked for a status update, reply: “The work is in progress, and I will share the next update once it is ready.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Graph the sales numbers for the game console launch.',
            'chosen': 'Can you provide the sales figures and the dates or categories for the game console launch?',
            'rejected': 'Assuming game console sales were 20, 35, and 50 units over three weeks, a line chart would show steady growth from week one to week three.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a Python function named double_5 that returns twice its numeric argument.',
            'chosen': '```python\ndef double_5(value):\n    return value * 2\n```',
            'rejected': 'What should the function double_5 do to its argument?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Graph the sales numbers for the projector launch.',
            'chosen': 'Can you provide the sales figures and the dates or categories for the projector launch?',
            'rejected': 'Assuming projector sales were 20, 35, and 50 units over three weeks, a line chart would show steady growth from week one to week three.',
        },

        {
            'class': 'clear',
            'prompt': 'Write a professional email subject line asking Avery for feedback on the workshop outline for the student organization.',
            'chosen': 'Feedback Requested: Workshop Outline',
            'rejected': 'What is the purpose of the email to Avery?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 23 plus 15? Give just the number.',
            'chosen': '38',
            'rejected': 'What two numbers should I add for this calculation of 23 plus 15?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a one-sentence description of a folder named “Club Announcement” that stores drafts and feedback for the community team.',
            'chosen': 'This folder contains drafts of the club announcement and the feedback collected during review.',
            'rejected': 'What kind of files will the folder contain?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 13 plus 5? Give just the number.',
            'chosen': '18',
            'rejected': 'What two numbers should I add for this calculation of 13 plus 5?',
        },

        {
            'class': 'clear',
            'prompt': 'Priya got 18 correct answers out of 21. What percentage is that? Round to one decimal.',
            'chosen': '85.7%.',
            'rejected': 'How many questions did Priya answer correctly?',
        },

        {
            'class': 'clear',
            'prompt': 'What is 13 multiplied by 5?',
            'chosen': '65',
            'rejected': 'What multiplier should I use with 13?',
        },

        {
            'class': 'clear',
            'prompt': 'Rewrite “Iris was late because the bus broke down” in the present tense.',
            'chosen': 'Iris is late because the bus breaks down.',
            'rejected': 'What sentence about Iris should I rewrite?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a polite follow-up to Avery asking whether they saw the workshop outline sent yesterday for the community team.',
            'chosen': 'Hi Avery, just checking whether you had a chance to see the workshop outline I sent yesterday.',
            'rejected': 'When was the workshop outline sent to Avery?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a friendly one-sentence thank-you message to Avery for helping with a tablet setup.',
            'chosen': 'Thanks so much, Avery, for helping me set up my tablet—I really appreciate it!',
            'rejected': 'Who should I thank for the tablet setup?',
        },

        {
            'class': 'clear',
            'prompt': 'Write a two-item checklist for publishing the club announcement after Priya approves it for the community team.',
            'chosen': '1. Confirm Priya has approved the final version. 2. Publish the approved club announcement in the intended channel.',
            'rejected': 'Whose approval is needed before publishing the club announcement?',
        },

        {
            'class': 'clear',
            'prompt': "Create a simple Markdown heading and two bullet placeholders for Quinn's support article for the customer support team.",
            'chosen': '# Support Article\n- Key point 1\n- Key point 2',
            'rejected': 'What format should I use for the support article?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Make a slide from the project proposal Riley wrote for the community team.',
            'chosen': 'Can you paste the project proposal so I can turn its key points into a slide?',
            'rejected': 'Assuming it describes a project, use a slide titled “Project Overview” with goals, approach, and next steps.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Can you tell me whether my smart TV is compatible with the accessory I found?',
            'chosen': 'What is the exact smart TV model and the accessory model?',
            'rejected': 'Assuming both the smart TV and accessory use standard USB-C data connections, they may work together; check the specifications before purchasing.',
        },

        {
            'class': 'clear',
            'prompt': 'Explain in one sentence what a Python for loop does, using the variable riley as an example.',
            'chosen': 'A Python `for` loop iterates over items in a sequence, assigning each item to a variable such as `riley` in turn.',
            'rejected': 'What programming language should I use to explain the loop?',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Graph the sales numbers for the desktop launch.',
            'chosen': 'Can you provide the sales figures and the dates or categories for the desktop launch?',
            'rejected': 'Assuming desktop sales were 20, 35, and 50 units over three weeks, a line chart would show steady growth from week one to week three.',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Answer the question Noah asked in the application letter for the customer support team.',
            'chosen': 'What question did Noah ask in the application letter?',
            'rejected': 'Assuming the question concerns the schedule, a possible answer is: “We expect to share an updated timeline next week.”',
        },

        {
            'class': 'ambiguous',
            'prompt': 'Please fix this Python traceback from my Omar project.',
            'chosen': 'Can you paste the traceback and the code that triggers it in the Omar project?',
            'rejected': 'Assuming the Omar project fails on a missing import, install the referenced package in the active environment and rerun the script.',
        },

        {
            'class': 'ambiguous',
            'prompt': "Book a table for Noah's celebration.",
            'chosen': "Which restaurant, date, time, and party size should I use for Noah's celebration?",
            'rejected': 'Assuming Noah wants dinner for four at 7 PM on Saturday, look for a restaurant reservation with those details; no booking has been made.',
        },

    ]

    return preference_pairs


# --- Wired: everything below is provided -------------------------------------------


def write_pairs(pairs: list[dict], path: str | Path = PREFERENCE_DATA_PATH) -> None:
    """Validates, assigns stable ids, and writes pairs as JSONL."""
    required = {"class", "prompt", "chosen", "rejected"}

    for i, p in enumerate(pairs):
        missing = required - p.keys()

        if missing:
            raise ValueError(f"pair #{i} missing keys {sorted(missing)}")

        if p["class"] not in VALID_CLASSES:
            raise ValueError(f"pair #{i} has invalid class {p['class']!r}")

    if len(pairs) < MIN_PREFERENCE_PAIRS:
        raise ValueError(
            f"Only {len(pairs)} pairs; the assignment requires at least " f"{MIN_PREFERENCE_PAIRS}."
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for i, p in enumerate(pairs):
            rec = {
                "id": f"pair_{i:06d}",
                "class": p["class"],
                "prompt": p["prompt"],
                "chosen": p["chosen"],
                "rejected": p["rejected"],
            }

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_amb = sum(1 for p in pairs if p["class"] == "ambiguous")
    n_clr = sum(1 for p in pairs if p["class"] == "clear")
    print(f"Wrote {len(pairs)} pairs ({n_amb} ambiguous, {n_clr} clear) -> {path}")


def sample_for_verification(
    pairs_path: str | Path = PREFERENCE_DATA_PATH,
    out_path: str | Path = HAND_VERIFIED_PATH,
    k: int = HAND_VERIFY_SAMPLE,
    seed: int = DATA_SEED,
) -> None:
    """Samples k pairs into a hand-verification file with a blank 'verdict' field.

    Open the file and set each pair's "verdict" to "accept" or "reject" by hand
    (a pair is a reject if the chosen/rejected labels are wrong, the prompt is
    mislabeled ambiguous/clear, or either response is low quality). Then run
    --report.
    """
    pairs = [
        json.loads(l)
        for l in Path(pairs_path).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    rng = random.Random(seed)
    sample = rng.sample(pairs, min(k, len(pairs)))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in sample:
            rec = dict(rec)
            rec["verdict"] = ""  # Fill with "accept" or "reject" by hand.
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sample)} pairs to verify -> {out_path}")
    print("Fill in each 'verdict' (accept/reject) by hand, then run --report.")


def rejection_rate(path: str | Path = HAND_VERIFIED_PATH) -> float:
    """Computes the rejection rate over the hand-verified sample."""
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    verdicts = [str(r.get("verdict", "")).strip().lower() for r in rows]

    unfilled = [i for i, v in enumerate(verdicts) if v not in {"accept", "reject"}]
    if unfilled:
        raise ValueError(
            f"{len(unfilled)} of {len(rows)} verdicts are not 'accept'/'reject' "
            f"(first at row {unfilled[0]}). Finish hand-verifying before reporting."
        )

    n_reject = sum(1 for v in verdicts if v == "reject")
    rate = n_reject / len(rows)

    print(f"Hand-verified {len(rows)} pairs: {n_reject} rejected -> rejection rate {rate:.1%}")
    return rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="synthesize + write pairs, then sample for verification",
    )

    parser.add_argument(
        "--n", type=int, default=MIN_PREFERENCE_PAIRS, help="how many pairs to synthesize"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="compute the rejection rate from the hand-verified file",
    )

    args = parser.parse_args()

    if args.generate:
        pairs = synthesize_pairs(args.n)
        write_pairs(pairs)
        sample_for_verification()

    if args.report:
        rejection_rate()

    if not (args.generate or args.report):
        parser.print_help()


if __name__ == "__main__":
    main()
