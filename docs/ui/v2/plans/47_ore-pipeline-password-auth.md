# Plan 47: Ore Pipeline Password Auth

Date: 2026-07-04

Spec: `docs/ui/v2/specs/ore-pipeline-password-auth-v0.1.md`

## Scope

Add optional Settings-managed password protection to `apps/ore_pipeline_web.py` and `apps/static/ore_pipeline_ui.html`.

## Implementation Steps

1. Extend app settings with an `auth` block.
   - Default: `password_enabled: false`.
   - Persist only PBKDF2-SHA256 hash metadata when a password is set.
   - Return only `password_enabled` in public settings payloads.

2. Add server auth helpers.
   - Verify password using constant-time hash comparison.
   - Create bounded in-memory session tokens.
   - Parse `Cookie` headers and validate session cookies.
   - Clear cookie on logout.

3. Gate requests.
   - Keep `/login`, `/api/auth/status`, `/api/auth/login`, and `/api/auth/logout` open.
   - Redirect unauthenticated UI pages to `/login`.
   - Return `401` for unauthenticated API/artifact access.

4. Add Settings UI.
   - Security panel with password field, remove-password checkbox, and status text.
   - Save sends `auth.password` only when the field is non-empty.
   - Save sends `auth.clear_password` only when the explicit remove checkbox is checked.
   - Reset-to-defaults preserves the existing password unless removal is explicit.

5. Test and document.
   - Unit-test settings hash persistence and public sanitization.
   - HTTP-test redirects, login cookie, protected API access, and clear-password behavior.
   - Update v2 docs index, user guide, session sync, changelog, and smoke checklist if applicable.
