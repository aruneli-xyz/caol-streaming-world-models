# Exact annotator instructions

You will see one generated scene view and a commanded transition label. Your
task is to identify the **first frame at which the commanded response becomes
visible in the scene**.

1. Read the commanded transition label.
2. Press **Play full clip**. Watch the entire clip once at normal speed (16
   frames per second), without changing tabs. Frame stepping and submission
   remain locked until full playback completes.
3. After full playback, replay as needed. Use Previous/Next to locate the exact
   first visible frame.
4. Choose exactly one response:
   - **Commanded response becomes visible at this frame**: select this only
     while the first visible response frame is displayed.
   - **No commanded response is visible**: use when it never becomes visible.
   - **Uncertain**: use when the evidence is genuinely ambiguous, including
     uncertainty about the exact onset.
   - **Technical failure**: use only for missing/corrupt frames, failed
     playback, or another tool problem that prevents judgment.
5. Rate confidence from 1 (very low) to 5 (very high), then submit.

Judge visible scene behavior, not what the model was instructed to do. For
**Forward → Backward**, look for the first visible transition toward backward
motion. For **Forward → Camera yaw right**, look for the first visible true
camera rotation to the right; do not treat sideways translation as yaw.

Do not infer an onset from the label alone. Do not compare against another
clip: no matched control is shown in this primary task. Some clips are hidden
quality checks or repeats; annotate every clip using the same instructions.
Do not refresh, inspect network requests, or share item contents. Report tool
problems to the study operator.
