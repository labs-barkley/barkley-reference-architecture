# Security Policy

The Barkley Reference Architecture is a **non-production research demonstrator**. It
processes **only synthetic, locally generated data**, makes **no network calls**,
connects to **no external service or API**, and handles **no real, personal, or
sensitive data**. It is **not** a deployed system and is **not diagnostic**.

As such, it has effectively no runtime security surface.

If you nonetheless find a security-relevant issue - for example, an unsafe pattern
that could matter if the code were adapted elsewhere - please report it privately to
**labs@getbarkley.com** rather than opening a public issue. Reports will be
acknowledged and addressed as the project's solo-maintainer capacity allows.