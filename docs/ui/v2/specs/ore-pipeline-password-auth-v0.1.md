# Ore Pipeline Password Auth v0.1

Date: 2026-07-04

## Goal

Add optional password protection to the v2 ore pipeline browser UI. When no password is set, the application behaves exactly as before. When a password is set in Settings, users must authenticate before opening UI pages, REST API endpoints, or run artifacts.

## User-Facing Behavior

- Settings includes a Security section with a Password field.
- Entering a non-empty password and saving settings enables password protection.
- If password protection is enabled, Settings shows that the password is set but never displays the password or password hash.
- Settings includes an explicit option to remove password protection.
- Resetting ordinary settings to defaults must not silently remove an existing password.
- A protected request without a valid session:
  - redirects browser UI pages to `/login`;
  - returns `401` JSON for REST/API/artifact requests.
- `/login` accepts the configured password and then returns the user to the requested page.
- Logout is available through the auth API, but no visible logout button is required for v0.1.

## Security Contract

- Store only a salted password hash in `outputs/ore_pipeline_ui/settings/app_settings.json`.
- Use PBKDF2-SHA256 with per-password random salt.
- Do not expose password hash, salt, or iterations through `/api/settings`.
- Authenticated sessions are cookie-based, `HttpOnly`, `SameSite=Lax`, path-wide, and bounded by an expiry.
- Sessions may be invalidated by service restart; that is acceptable for the local GUI.
- This is local/VM access control, not a full identity system. It does not replace TLS or network firewall rules on exposed deployments.

## API Contract

- `GET /api/auth/status`
  - Always accessible.
  - Returns `{password_enabled, authenticated}`.
- `POST /api/auth/login`
  - Always accessible.
  - Body: `{password: string}`.
  - On success returns `{ok: true, authenticated: true}` and sets the session cookie.
  - On failure returns `401`.
- `POST /api/auth/logout`
  - Always accessible.
  - Clears the session cookie and returns `{ok: true}`.
- `GET /api/settings`
  - Protected when password protection is enabled.
  - Returns `auth: {password_enabled: boolean}` only.
- `PUT /api/settings`
  - Protected when password protection is enabled.
  - Accepts `auth.password` to set/change the password.
  - Accepts `auth.clear_password: true` to remove password protection.

## Non-Goals

- No user accounts or roles.
- No password recovery.
- No persistent browser localStorage password.
- No TLS termination inside this app.
