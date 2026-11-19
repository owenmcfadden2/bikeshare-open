---
name: seed-fault
description: Deliberately inject one described fault into src/, record the hook stderr and red test output in docs/SEEDED_FAULTS.md, then stop without fixing. Use when asked to seed, inject, or simulate a fault.
allowed-tools: Read, Edit, Bash(make test*), Bash(python -m pytest*), Bash(git diff*), Bash(git status*)
---
# Current state of src/ and tests/
!`git status --short src/ tests/`

# Seeding fault: $ARGUMENTS

1. **Precondition.** The status above must be empty. If it lists any
   file, stop and say so. Run `make test`; if anything fails, stop.
2. **Make exactly one change** in src/ that implements the fault above.
   Use the Edit tool, never Bash, so the PostToolUse hook runs. Do not
   touch tests/ or data/.
3. **Show the evidence verbatim:** the output of `git diff src/`; the
   hook's stderr exactly as received, or the sentence "The hook did not
   fire."; then the full output of `make test`. Never summarize output.
4. **Append a section to docs/SEEDED_FAULTS.md** headed with the fault
   description, containing those three outputs pasted, not described.
5. **Stop.** Do not fix the fault. The fix happens in a fresh session so
   the turns it needs can be counted. To abandon instead:
   `git checkout -- src/`.

If no test goes red, say so plainly in step 4. A fault the suite cannot
see is a finding about the suite, not a failure of this skill.
