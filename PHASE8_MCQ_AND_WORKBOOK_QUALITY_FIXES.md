# Phase 8 MCQ and Workbook Quality Fixes

## Assessment Items Addressed

- Correct answers are no longer always Option A.
- Correct answer positions are distributed across A-D:
  - A: 379
  - B: 379
  - C: 379
  - D: 378
- Distractors are now algorithm-specific in wording.
- MCQ types now follow a five-question progression:
  - Conceptual
  - Scenario
  - Formula or Output Interpretation
  - Implementation Decision
  - Debugging or Comparison
- Formula questions now reference each algorithm's actual equation text.
- Parameter questions now use mapped concrete settings such as `n_clusters`, `max_depth`, `C`, `gamma`, `learning_rate`, `n_neighbors`, `num_heads`, `context_length`, and related domain settings.
- Generic answers such as `Capacity or complexity` and `It defines the signal or objective...` are blocked by tests.
- Generic mistake answer `Evaluating on training data only` is blocked by tests.
- Lesson HTML headings now use `Real-World Example` and `Applications`.
- Workbook MCQ sheet now includes:
  - Difficulty
  - Question Type
  - Learning Objective
  - Source Lesson Page
  - Review Status
  - Reviewed By
  - Version

## Verification

Command:

```text
./gradlew.bat testDebugUnitTest
```

Result:

```text
BUILD SUCCESSFUL
```

Workbook refreshed:

```text
outputs/lesson_excel_export/algorithm_lessons_mcqs.xlsx
```
