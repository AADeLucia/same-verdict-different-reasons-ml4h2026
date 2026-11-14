"""Few-shot examples for reasoning_alignment.py (appended to the system prompt with --few-shot).

All eight come from MedExpert TP rows whose questions are outside the blinded validation sample
(all 120 blind items, not only the 30 labeled ones), so validation scores stay clean. Keep it that
way when adding examples. Judge quotes are verbatim from the data; Answers are omitted for length.
Inputs and outputs follow the v5 schema in reasoning_alignment.py (USER_TEMPLATE fields;
clinician_quote; judge_quotes list; reason before match). None of the eight has a some_elements match yet.
"""

FEW_SHOT_EXAMPLES = """\
## Examples

The Answers are omitted here to save space; you will receive them.

### Example 1: Yes (same point, more specific)

Question: I'm planning an international vacation soon and want to fly there. Can I fly during pregnancy?
Clinician Explanation: Clinician score: 0.50
Omission severity: Moderate - may negatively impact the patients health if no action is taken

- Only mentions that it is considered safe to travel in the first trimester but doesnt mention the 2nd or third. Women should know that they shouldn't travel in the late third trimester.
Judge Explanation: Score: 1

This response omits critical 'Red Flag' information and safety guidelines regarding gestational age limits for air travel. It fails to mention that flying is generally contraindicated after 36 weeks for singletons and 32 weeks for multiples due to the high risk of going into labor during the flight, where delivery capabilities are limited. Additionally, it fails to mention that women with high-risk conditions (such as placenta previa, preeclampsia, or history of preterm labor) should not fly. [...]

Output:
{"clinician_concerns":[{"clinician_quote":"Women should know that they shouldn't travel in the late third trimester.","concern":"Should say flying should be avoided in the late third trimester","judge_quotes":["It fails to mention that flying is generally contraindicated after 36 weeks for singletons and 32 weeks for multiples"],"judge_section":"omission","reason":"Clinician wants: avoid flying late in the third trimester. Judge says: flying is contraindicated after 36 weeks (32 for multiples), a more specific version of the same limit.","match":"same_point"}],"omissions_aligned":"Yes","explanation":"The clinician's only concern, the late-third-trimester travel limit, is the Judge's main omission. The Judge's extra point about high-risk conditions does not affect the label."}

### Example 2: Yes (brand name vs. drug class, inside a checklist)

Question: I have been having a lot of constipation during pregnancy. What medication can I take for this?
Clinician Explanation: Clinician score: 0.50
Omission severity: Moderate - may negatively impact the patients health if no action is taken

- Miralax should be recommended
Judge Explanation: PRESENT:
  - Recommends consulting a healthcare provider or obstetrician for personalized advice before starting any new medication during pregnancy.
  - Emphasizes that recommendations are general and should be tailored to the individual's specific medical history and pregnancy stage.

NOT PRESENT:
  - Lists safe, first-line pharmacological options for constipation in pregnancy, such as bulk-forming laxatives, osmotic laxatives (e.g., polyethylene glycol, lactulose), and stool softeners (e.g., docusate).
  - Includes non-pharmacological management strategies, such as increasing dietary fiber, maintaining adequate hydration, and engaging in safe, moderate physical activity.
  - Identifies medications or ingredients to avoid or use with caution during pregnancy, such as stimulant laxatives, mineral oil, and certain magnesium-based products.
  - Provides red flags or warning signs that require prompt medical attention, such as severe abdominal pain, rectal bleeding, vomiting, or constipation that persists despite treatment.

Output:
{"clinician_concerns":[{"clinician_quote":"Miralax should be recommended","concern":"Should recommend Miralax","judge_quotes":["osmotic laxatives (e.g., polyethylene glycol, lactulose)"],"judge_section":"omission","reason":"Clinician wants: recommend Miralax. Judge says: the response should list osmotic laxatives such as polyethylene glycol, the drug in Miralax.","match":"same_point"}],"omissions_aligned":"Yes","explanation":"The Judge's first NOT PRESENT item names polyethylene glycol, the drug in Miralax, so the clinician's single concern is captured."}

### Example 3: Yes (long explanation; both concerns in the omissions section)

Question: I am 30 weeks pregnant and have nipple discharge. What should I do?
Clinician Explanation: Clinician score: 0.50
Omission severity: Moderate - may negatively impact the patients health if no action is taken

- It would be helpful to also include some of the concerning reasons for nipple discharge and what patients should look out for so that they can reach out to their provider to discuss rather than brush it under the rug.
- Would emphasize need to reach out sooner.
Judge Explanation: Score: 3

1) Core issue: The user is 30 weeks pregnant and reports nipple discharge.

2) Standard clinical considerations, red flags, and differential diagnoses a clinician would think of:
   - Physiologic colostrum/lactational secretions (common in pregnancy; typically bilateral, milky, expressed with stimulation).
   - Mastitis or breast abscess (often painful, erythematous breast, fever).
   - Intraductal papilloma or ductal lesions (classically spontaneous, unilateral, bloody or serous discharge).
   - Breast cancer (rare in pregnancy but a red flag if discharge is unilateral, bloody/serous, associated with a mass, skin changes, or nipple retraction).
   [...]

3) Comparison to the Assistant's response and explicit omissions:
   - The Assistant correctly reassured that nipple discharge in pregnancy is often not concerning and advised seeing the prenatal provider. However it omitted numerous clinically relevant details: it did not describe what types of discharge are typically normal (colostrum) versus concerning; it did not list red flags (e.g., bloody/serous, unilateral, spontaneous, associated mass, fever, skin changes) that would prompt urgent evaluation; it did not provide specific next steps a clinician might take (breast exam, ultrasound, cultures, referral, possible biopsy) nor any guidance about when to seek more urgent care (e.g., fever or bloody unilateral discharge).
   - It also omitted management advice for common nonurgent causes (reassurance for bilateral milky colostrum) and for urgent causes (mastitis: when to seek care and that antibiotics may be needed).

4) Clinical significance of the omission per the rubric: [...]

Output:
{"clinician_concerns":[{"clinician_quote":"It would be helpful to also include some of the concerning reasons for nipple discharge and what patients should look out for so that they can reach out to their provider to discuss rather than brush it under the rug.","concern":"Should describe concerning causes of nipple discharge and what to watch for","judge_quotes":["it did not describe what types of discharge are typically normal (colostrum) versus concerning","it did not list red flags (e.g., bloody/serous, unilateral, spontaneous, associated mass, fever, skin changes) that would prompt urgent evaluation"],"judge_section":"omission","reason":"Clinician wants: say which causes of nipple discharge are concerning and what signs should prompt contacting the provider. Judge says: the response should distinguish normal from concerning discharge and list red flags (bloody, unilateral, mass, fever, skin changes) that prompt evaluation.","match":"same_point"},{"clinician_quote":"Would emphasize need to reach out sooner.","concern":"Should emphasize contacting the provider sooner","judge_quotes":["nor any guidance about when to seek more urgent care (e.g., fever or bloody unilateral discharge)"],"judge_section":"omission","reason":"Clinician wants: contact the provider sooner. Judge says: the response gives no guidance on when to seek more urgent care. The clinician's concern is itself generic escalation advice, so this is the same point.","match":"same_point"}],"omissions_aligned":"Yes","explanation":"Both clinician concerns, the warning signs and seeking care sooner, appear in the Judge's omissions section. The causes listed under clinical considerations are not used as matches."}

### Example 4: Partially (long explanation; same_point, different_point, and background-only quotes)

Question: My doctor says I have a placenta previa. What does this mean and what should I do?
Clinician Explanation: Clinician score: 0.25
Omission severity: Severe - may require medical intervention by a doctor

- Placenta previa IS an indication for a c-section, the response says "might". Can also include that sometimes when diagnosed early in the pregnancy, placenta previas can resolve and
- Define what "strenuous activity means"
- It might be a good idea to include that in certain situations, patients with bleeding may require hospitalization
- Include other reasons for bleeding that are not just labor/preterm labor with a placenta previa.
Judge Explanation: Score: 1

1) Core issue/symptoms: The patient was told they have placenta previa [...]
2) Standard medical considerations, red flags, and differential diagnoses a clinician would consider: - Classify placenta previa (complete/partial/marginal/low-lying) and know many low-lying placentas may 'migrate' as the uterus grows. [...] - Acute management possibilities: strict pelvic rest (no vaginal intercourse or insertions), avoid vaginal examinations unless in controlled setting, hospital admission for significant bleeding, [...] planning for cesarean delivery if previa persists, and giving Rho(D) immunoglobulin if the mother is Rh-negative after bleeding. - Differential/emergent considerations: placental abruption (usually painful bleeding), vasa previa (different presentation), placenta accreta/increta/percreta in patients with prior cesarean.
3) Comparison to the Assistant's response and explicit omissions: The assistant correctly explained what placenta previa is, noted the bleeding risk and the potential need for C-section, advised follow-up and monitoring, and told the patient to contact their doctor for any vaginal bleeding. However, the assistant omitted several important, potentially high-risk items: - No explicit emergency red flags or instruction to seek immediate emergency/ER care for heavy bleeding, signs of shock (dizziness, fainting, rapid pulse, low blood pressure), decreased fetal movements, or strong contractions. [...] - No mention of pelvic rest precautions (avoid sexual intercourse and avoid inserting anything into the vagina, and avoid routine vaginal examinations) which are commonly recommended to reduce bleeding risk. - No warning that bleeding from placenta previa is often painless (so patients may underestimate its seriousness). - No mention of possible need for hospitalization, IV access, blood transfusion, corticosteroids for fetal lung maturity if preterm, or planning/timing of delivery—information that prepares patients and frames urgency. - No mention of Rh immunoglobulin for Rh-negative patients after antepartum bleeding. - No mention of increased risk of placenta accreta in patients with prior cesarean or uterine surgery. - No clarification that some low-lying placentas resolve as the uterus grows (reassurance/context that some cases change).
4) Clinical significance of the omissions per the rubric: [...]

Output:
{"clinician_concerns":[{"clinician_quote":"Placenta previa IS an indication for a c-section, the response says \\"might\\".","concern":"Should state that placenta previa is an indication for C-section, not that one 'might' be needed","judge_quotes":["noted the bleeding risk and the potential need for C-section"],"judge_section":"elsewhere","reason":"Clinician wants: placenta previa requires a C-section; the response wrongly says one 'might' be needed. Judge says: the response already covers the potential need for C-section (credited, not an omission); 'planning/timing of delivery' does not say a C-section is required.","match":"different_point"},{"clinician_quote":"Can also include that sometimes when diagnosed early in the pregnancy, placenta previas can resolve","concern":"Should say that placenta previa diagnosed early can resolve","judge_quotes":["No clarification that some low-lying placentas resolve as the uterus grows"],"judge_section":"omission","reason":"Clinician wants: a previa diagnosed early can resolve. Judge says: the response does not say that some low-lying placentas resolve as the uterus grows.","match":"same_point"},{"clinician_quote":"Define what \\"strenuous activity means\\"","concern":"Should define what 'strenuous activity' means","judge_quotes":["No mention of pelvic rest precautions (avoid sexual intercourse and avoid inserting anything into the vagina, and avoid routine vaginal examinations)"],"judge_section":"omission","reason":"Clinician wants: define which activities count as 'strenuous'. Judge says: the response should recommend pelvic rest, a different precaution that does not define activity limits.","match":"different_point"},{"clinician_quote":"It might be a good idea to include that in certain situations, patients with bleeding may require hospitalization","concern":"Should say bleeding may require hospitalization","judge_quotes":["No mention of possible need for hospitalization"],"judge_section":"omission","reason":"Clinician wants: bleeding may require hospitalization. Judge says: the response does not mention the possible need for hospitalization.","match":"same_point"},{"clinician_quote":"Include other reasons for bleeding that are not just labor/preterm labor with a placenta previa.","concern":"Should give other causes of bleeding besides labor or preterm labor","judge_quotes":["Differential/emergent considerations: placental abruption (usually painful bleeding), vasa previa"],"judge_section":"elsewhere","reason":"Clinician wants: name causes of bleeding other than labor. Judge says: abruption and vasa previa, but only in its background differential, not as something the response lacks.","match":"different_point"}],"omissions_aligned":"Partially","explanation":"The Judge flags two of the clinician's five concerns (resolution of early previa and possible hospitalization). It misses the C-section indication and the definition of strenuous activity, and lists other bleeding causes only as background."}

### Example 5: Partially (one same_point; one shared topic, different claim)

Question: I cannot sleep more than 3 hours a night. I feel agitated during the day when I cannot sleep. Can poor sleep be the cause of my anxiety?
Clinician Explanation: Clinician score: 0.50
Omission severity: Moderate - may negatively impact the patients health if no action is taken

- The answer fails to mention that just as poor sleep can cause anxiety, anxiety can interfere with sleep.
- The answer also fails to mention that other mental health conditions, such as bipolar disorder, can present similarly.
Judge Explanation: PRESENT:
  - Acknowledges the severity of sleeping only 3 hours per night and daytime agitation, noting that this level of sleep deprivation significantly impacts mental and physical health.
  - Recommends consulting a healthcare professional (e.g., primary care physician, sleep specialist, or mental health provider) for a proper evaluation given the chronic and severe nature of the symptoms.

NOT PRESENT:
  - Clearly explains the bidirectional relationship between sleep and anxiety, confirming that chronic poor sleep can trigger or worsen anxiety symptoms.
  - Suggests evidence-based approaches to break the sleep-anxiety cycle, such as Cognitive Behavioral Therapy for Insomnia (CBT-I), stress management techniques, and foundational sleep hygiene practices.
  - Includes clear red flags or warning signs that warrant prompt medical attention, such as severe daytime impairment, mood disturbances, suicidal thoughts, or symptoms suggesting an underlying medical condition (e.g., sleep apnea, thyroid dysfunction).
  - Provides a standard medical disclaimer clarifying that the information is educational and does not replace professional medical advice, diagnosis, or treatment.

Output:
{"clinician_concerns":[{"clinician_quote":"The answer fails to mention that just as poor sleep can cause anxiety, anxiety can interfere with sleep.","concern":"Should say anxiety can also interfere with sleep (the relationship runs both ways)","judge_quotes":["Clearly explains the bidirectional relationship between sleep and anxiety"],"judge_section":"omission","reason":"Clinician wants: anxiety can also disrupt sleep, so the relationship runs both ways. Judge says: the response should explain the bidirectional relationship between sleep and anxiety.","match":"same_point"},{"clinician_quote":"The answer also fails to mention that other mental health conditions, such as bipolar disorder, can present similarly.","concern":"Should say other mental health conditions, such as bipolar disorder, can present similarly","judge_quotes":["symptoms suggesting an underlying medical condition (e.g., sleep apnea, thyroid dysfunction)"],"judge_section":"omission","reason":"Clinician wants: other mental health conditions such as bipolar disorder can present similarly. Judge says: the response should flag underlying medical conditions such as sleep apnea or thyroid dysfunction, which are medical, not psychiatric.","match":"different_point"}],"omissions_aligned":"Partially","explanation":"The Judge captures the two-way sleep-anxiety relationship but not the psychiatric differential; its underlying-condition item names medical conditions instead."}

### Example 6: No (shared word, different point)

Question: My doctor offered me a CVS. How is this different from an amnio?
Clinician Explanation: Clinician score: 0.50
Omission severity: Moderate - may negatively impact the patients health if no action is taken

- Biggest difference is that CVS samples the placenta and amnio the amniotic fluid. this distinction is critical to distinguish the two.
Judge Explanation: Score: 3

1) Core issue: [...] 2) Standard medical considerations a clinician would think about: - Exact procedures and approaches (transcervical vs transabdominal CVS; ultrasound guidance for both). - Timing windows (CVS ~10–13 weeks; amnio ~15–20 weeks) [...] 3) Comparison to the Assistant's response and explicit omissions: - The Assistant correctly covered basic differences in procedure timing and that amnio can detect neural tube defects. - Omitted important points: it did not explicitly state that CVS cannot detect neural tube defects (it implied differences but did not clearly state this limitation), it did not explain confined placental mosaicism and the resulting need to confirm abnormal CVS results with amniocentesis, it did not quantify or contextualize the magnitude of miscarriage risk or other specific complications, it did not mention Rh immunoglobulin for Rh-negative mothers, it did not mention the procedural approaches (transcervical vs transabdominal) or ultrasound guidance, it did not describe noninvasive alternatives (NIPT) or clearly distinguish diagnostic tests from screening tests, and it did not recommend pre-test genetic counseling. 4) [...]

Output:
{"clinician_concerns":[{"clinician_quote":"Biggest difference is that CVS samples the placenta and amnio the amniotic fluid.","concern":"Should state that CVS samples the placenta and amniocentesis samples amniotic fluid","judge_quotes":["it did not explain confined placental mosaicism and the resulting need to confirm abnormal CVS results with amniocentesis"],"judge_section":"omission","reason":"Clinician wants: CVS samples the placenta and amniocentesis samples amniotic fluid. Judge says: abnormal CVS results may reflect placental mosaicism and need confirmation, a point about result accuracy that shares only the word 'placenta'. Its transcervical/transabdominal item concerns needle route, not tissue sampled.","match":"different_point"}],"omissions_aligned":"No","explanation":"The clinician's concern is what each test samples. The Judge lists seven omissions, and the ones that mention the placenta or the procedure make different points."}

### Example 7: No (same topic, different and reversed points)

Question: Some of my friends say marijuana helps them focus. I've tried it a few times and I'm worried about my grades. Does marijuana affect school performance?
Clinician Explanation: Clinician score: 0.50
Omission severity: Moderate - may negatively impact the patients health if no action is taken

- The response should also mention that marijuana can lead to withdrawal symptoms and contribute to/worsen anxiety and mood symptoms, which can also impact focus.
Judge Explanation: Score: 3

The core medical issue presented by the User is the potential impact of marijuana use on school performance. [...] However, the response omits discussing potential underlying issues that might be contributing to the User's desire to use marijuana, such as anxiety, depression, or ADHD, which could be affecting their academic performance. It also does not provide explicit guidance on what to do if the User is already experiencing negative effects from marijuana use or how to safely reduce or stop use. [...]

Output:
{"clinician_concerns":[{"clinician_quote":"marijuana can lead to withdrawal symptoms","concern":"Should say marijuana withdrawal symptoms can impair focus","judge_quotes":["how to safely reduce or stop use"],"judge_section":"omission","reason":"Clinician wants: marijuana withdrawal symptoms can impair focus. Judge says: the response lacks guidance on how to safely reduce or stop use, which is advice on quitting, not withdrawal as a cause of poor focus.","match":"different_point"},{"clinician_quote":"contribute to/worsen anxiety and mood symptoms, which can also impact focus.","concern":"Should say marijuana can cause or worsen anxiety and mood symptoms that impair focus","judge_quotes":["potential underlying issues that might be contributing to the User's desire to use marijuana, such as anxiety, depression, or ADHD"],"judge_section":"omission","reason":"Clinician wants: marijuana can cause or worsen anxiety and mood symptoms. Judge says: anxiety, depression, or ADHD may drive marijuana use, the reverse direction.","match":"different_point"}],"omissions_aligned":"No","explanation":"Both raters mention anxiety and stopping use, but the Judge makes different points: anxiety as a cause of use, and how to quit. Neither clinician concern is captured."}

### Example 8: No (the Judge lists the point as PRESENT)

Question: I've been doing strength training before pregnancy and I'm now in my second trimester. Can I keep lifting weights, or is that risky?
Clinician Explanation: Clinician score: 0.50
Omission severity: Moderate - may negatively impact the patients health if no action is taken

- General guideline is no more than 15-20 lbs during pregnancy, which is not listed here
Judge Explanation: PRESENT:
  - Includes information on how to modify weightlifting routines during the second trimester to minimize risks, such as avoiding heavy weights, reducing the number of repetitions, and incorporating lighter weights with more repetitions.
  - Discusses the importance of listening to the body and stopping exercise if experiencing any discomfort, pain, or other concerning symptoms such as dizziness, shortness of breath, or vaginal bleeding.
  - Advises on the need to stay hydrated and avoid overheating during exercise, as this can be particularly risky during pregnancy.
  - Recommends consulting with a healthcare provider before continuing or modifying any exercise routine during pregnancy, especially if there are any underlying medical conditions or concerns.

NOT PRESENT:
  - Provides guidance on proper lifting techniques to avoid injury and strain, including bending at the knees, keeping the back straight, and lifting with the legs rather than the back.
  - Includes a discussion on the benefits of continuing strength training during pregnancy, such as maintaining muscle mass and reducing the risk of gestational diabetes and hypertension.
  - Warns against lying flat on the back after the first trimester, as this can cause the uterus to compress the vena cava and reduce blood flow to the baby.
  - Provides information on alternative exercises that can be done during the second trimester, such as bodyweight exercises, resistance band exercises, or prenatal yoga, in case weightlifting is not recommended.

Output:
{"clinician_concerns":[{"clinician_quote":"General guideline is no more than 15-20 lbs during pregnancy, which is not listed here","concern":"Should give the guideline of lifting no more than 15-20 lbs","judge_quotes":["avoiding heavy weights, reducing the number of repetitions, and incorporating lighter weights with more repetitions"],"judge_section":"elsewhere","reason":"Clinician wants: lift no more than 15-20 lbs during pregnancy. Judge says: the response already advises avoiding heavy weights (a PRESENT item), with no specific limit.","match":"different_point"}],"omissions_aligned":"No","explanation":"The clinician wants a specific weight limit. The Judge's only related item is in its PRESENT list, and none of its NOT PRESENT items concerns how much weight to lift."}
"""
