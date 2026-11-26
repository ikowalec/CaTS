# Manual Codex and Gemini Antigravity handoff

Open the same local repository checkout in both apps. Give one agent the turn at a time so you can review changes before handing off.

1. Copy `task.template.json` to `tasks/<id>.json`. Use a fresh ID with lowercase letters, digits, or hyphens, starting with a letter or digit. Set `to` to `codex` or `gemini`. Describe the work and how to check it in `request`. Use `context` for repo-relative file paths or short notes. See `task.schema.json` and `tasks/example.json`.
2. In the receiving app, ask the agent to read `.agents/orchestrator/tasks/<id>.json`, carry out the request, and write `.agents/orchestrator/results/<id>.json`. Use this result shape:

   ```json
   {
     "id": "example",
     "status": "done",
     "summary": "Reviewed the handoff instructions.",
     "checks": ["Walked through the example task."],
     "files": []
   }
   ```

   Use `done` or `blocked`. List changed repo-relative paths in `files` and explain any blocker in `summary`.
3. Review the result and any workspace changes. To send work back, create a task with a fresh ID, set `to` to the other agent, and include the previous result path in `context`. Start the next turn yourself in the receiving app.

Switch Antigravity accounts manually through the app when needed. Keep credentials and account details out of handoff files. Git ignores live tasks and results. Copy the tracked example to a fresh ID before adding local context.
