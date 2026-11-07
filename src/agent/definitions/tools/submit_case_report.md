---
name: submit_case_report
description: Submit the final case report. Every claim is checked by a program against the data.
args_schema: LLMNarrative
---
Call this exactly once when you have finished investigating. If the checker finds problems it
will tell you which claims failed and why; then fix or remove them and call this tool again.
