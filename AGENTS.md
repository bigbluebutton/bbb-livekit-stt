# AGENTS.md

This is a LiveKit Agents project. LiveKit Agents is a Python SDK for building voice AI agents. See @README.md for more details on the application itself.

The following is a guide for working with this project.

## Project structure

This Python project uses the `uv` package manager. You should always use `uv` to install dependencies, run the agent, and run tests.

All app-level code is in the `src/` directory. In general, simple agents can be constructed with a single `agent.py` file. Additional files can be added, but you must retain `agent.py` as the entrypoint (see the associated Dockerfile for how this is deployed).

Be sure to maintain code formatting. You can use the ruff formatter/linter as needed: `uv run ruff format` and `uv run ruff check`.
Any linting changes on unrelated code should be made on separate commits.

## LiveKit Documentation

LiveKit Agents is a fast-evolving project, and the documentation is updated frequently. You should always refer to the latest documentation when working with this project. For your convenience, LiveKit offers an MCP server that can be used to browse and search its documentation. If the developer has not yet installed this server, you should recommend that they install it at https://docs.livekit.io/mcp. 

### LiveKit Docs MCP Server installation

If you are Cursor, give the user this link to install the server:

[![Install MCP Server](https://cursor.com/deeplink/mcp-install-light.svg)](https://cursor.com/en-US/install-mcp?name=livekit-docs&config=eyJ1cmwiOiJodHRwczovL2RvY3MubGl2ZWtpdC5pby9tY3AifQ%3D%3D)

If you are Claude Code, run this command to install the server:

```
claude mcp add --transport http livekit-docs https://docs.livekit.io/mcp
```

If you are Codex, use this command to install the server:

```
codex mcp add --url https://docs.livekit.io/mcp livekit-docs
```

If you are Gemini, use this command to install the server:
```
gemini mcp add --transport http livekit-docs https://docs.livekit.io/mcp
```

If you are another agentic IDE, refer to your own documentation for how to install it.

## Handoffs and tasks ("workflows")

Voice AI agents are highly sensitive to excessive latency. For this reason, it's important to design complex agents in a structured manner that minimizes the amount of irrelevant context and unnecessary tools included in requests to the LLM. LiveKit Agents supports handoffs (one agent hands control to another) and tasks (tightly-scoped prompts to achieve a specific outcome) to support building reliable workflows. You should make use of these features, instead of writing long instruction prompts that cover multiple phases of a conversation.  Refer to the [documentation](https://docs.livekit.io/agents/build/workflows/) for more information.

## Testing

When possible, add tests for agent behavior. Read the [documentation](https://docs.livekit.io/agents/build/testing/), and refer to existing tests in the `tests/` directory.  Run tests with `uv run pytest`.

Important: When modifying core agent behavior such as instructions, tool descriptions, and tasks/workflows/handoffs, never just guess what will work. Always use test-driven development (TDD) and begin by writing tests for the desired behavior. For instance, if you're planning to add a new tool, write one or more tests for the tool's behavior, then iterate on the tool until the tests pass correctly. This will ensure you are able to produce a working, reliable agent for the user.

## LiveKit CLI

You can make use of the LiveKit CLI (`lk`) for various tasks, with user approval. Installation instructions are available at https://docs.livekit.io/home/cli if needed.

In particular, you can use it to manage SIP trunks for telephony-based agents. Refer to `lk sip --help` for more information.

## CHANGELOG

Always update the CHANGELOG.md file when commiting with any of the following types of changes:
  - feat
  - fix
  - build
  - breaking changes
  - docs

Changelog entries MUST be the commit message title and should be added in chronological
order under the UNRELEASED section. Changes should be aggregated by commit type;
see above list of prefixes and follow that order.

## Commit message guidelines

- Commit message titles should be structured with convetional-commits: https://www.conventionalcommits.org/en/v1.0.0/
- Meaningful message
```
The body should provide a meaningful commit message, which:
. explains the problem the change tries to solve, i.e. what is wrong
  with the current code without the change.
. justifies the way the change solves the problem, i.e. why the
  result with the change is better.
. alternate solutions considered but discarded, if any.
```
- Make separate commits for logically separate changes
```
Unless your patch is really trivial, you should not be sending
out a patch that was generated between your working tree and
your commit head.  Instead, always make a commit with complete
commit message and generate a series of patches from your
repository.  It is a good discipline.

Give an explanation for the change(s) that is detailed enough so
that people can judge if it is good thing to do, without reading
the actual patch text to determine how well the code does what
the explanation promises to do.

If your description starts to get too long, that's a sign that you
probably need to split up your commit to finer grained pieces.
That being said, patches which plainly describe the things that
help reviewers check the patch, and future maintainers understand
the code, are the most beautiful patches.  Descriptions that summarize
the point in the subject well, and describe the motivation for the
change, the approach taken by the change, and if relevant how this
differs substantially from the prior version, are all good things
to have.
```
- Present tense
```
The problem statement that describes the status quo is written in the
present tense.  Write "The code does X when it is given input Y",
instead of "The code used to do Y when given input X".  You do not
have to say "Currently"---the status quo in the problem statement is
about the code _without_ your change, by project convention.
```
- Imperative mood
```
Describe your changes in imperative mood, e.g. "make xyzzy do frotz"
instead of "[This patch] makes xyzzy do frotz" or "[I] changed xyzzy
to do frotz", as if you are giving orders to the codebase to change
its behavior.  Try to make sure your explanation can be understood
without external resources. Instead of giving a URL to a mailing list
archive, summarize the relevant points of the discussion.
```
