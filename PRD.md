# 📘 {{PROJECT_NAME}} — Product Requirements Document (PRD)

> Source of truth for **what** {{PROJECT_NAME}} is, **who** it serves, and **what done looks like**.
> Architecture (the **how**) lives in `PLANNING.md` (and optionally `ARCHITECTURE.md`).

---

## 1. Project Goals

### 1.1 MVP Goals (Phase 1)

A one-paragraph framing of the MVP outcome. Replace this paragraph with a concrete description of the **smallest end-to-end slice** that delivers value to the first user.

The MVP must:

- Deliver the {{primary user flow}} from start to finish.
- Expose a clear interface (REST API / CLI / UI) aligned with the modular architecture in `PLANNING.md`.
- Emit **domain events** for state changes that other modules might react to (now or later).
- Run as a single deployable unit with clear separation between modules.

### 1.2 Future Goals (Phase 2+)

Post-MVP goals preserve core stability while preparing for extraction or expansion:

- Extract {{candidate module}} as a standalone service consuming the same event contracts.
- Replace the in-process event bus with a transport-friendly bus (Redis/NATS/Kafka) without rewriting domain logic.
- Add: {{future capability 1}}, {{future capability 2}}, {{future capability 3}}.

### 1.3 Non-Goals

- {{non-goal 1: e.g., real-time chat}}
- {{non-goal 2: e.g., marketplace, payments}}
- {{non-goal 3: e.g., enterprise SSO}}
- Premature service split before module boundaries are proven.
- Third-party SaaS dependencies for core flows.

---

## 2. Scope — Core Features

| Area              | MVP Scope                                         |
| ----------------- | ------------------------------------------------- |
| **Auth**          | {{auth scope}}                                    |
| **{{Domain A}}**  | {{scope}}                                         |
| **{{Domain B}}**  | {{scope}}                                         |
| **{{Domain C}}**  | {{scope}}                                         |
| **Events**        | Internal event bus; canonical event names + payloads. |
| **Security**      | Authn + authz + input validation + error normalization. |
| **Deployment**    | Containerized; single `docker-compose.yml`; CI green on `main`. |

---

## 3. Constraints

### 3.1 Technical Constraints

- **Backend framework:** {{NestJS | Express | other}}.
- **Language:** TypeScript end-to-end (server, client, shared types).
- **Frontend:** {{React + Next.js | other}}.
- **Database:** {{PostgreSQL | MongoDB}}. One per deployment in MVP.
- **No external network libraries** beyond the core stack.
- **No external APIs** for core flows.
- **Internal event bus only** in MVP — no Kafka/RabbitMQ/Redis Pub/Sub.
- **Auth:** {{JWT | session}} only.
- **File handling:** Local filesystem in MVP; abstraction for S3-compatible swap later.

### 3.2 Architectural Constraints

- **Modular monolith** — domain modules MUST NOT cross-import logic. Communication via events or via internal services within the same module.
- **Service-oriented core:** Controllers thin, Services own business logic, Repositories own data access.
- **Derived state preference:** computed values are not stored as denormalized truth.
- **Folder structure** defined in `PLANNING.md` is binding.

### 3.3 Operational Constraints

- Single-instance deployment in MVP.
- No paid third-party infrastructure required to run dev or MVP prod.
- All secrets via environment variables; never committed.

---

## 4. User Stories & Acceptance Criteria

> Format: **As a `<role>`, I want `<capability>` so that `<value>`.**
> Acceptance criteria are testable, observable, and bounded.

---

### 4.1 {{Role A — e.g., Admin}}

#### US-A1 — {{Capability}}

**As a** {{Role A}}, **I want** {{capability}} **so that** {{value}}.

**Acceptance Criteria:**

- [ ] {{Endpoint or action}} performs {{behavior}}.
- [ ] {{Failure case}} returns {{specific error}}.
- [ ] {{Authorization rule}} is enforced.

#### US-A2 — {{Capability}}

(repeat the structure)

---

### 4.2 {{Role B — e.g., Teacher}}

#### US-T1 — {{Capability}}

**As a** {{Role B}}, **I want** {{capability}} **so that** {{value}}.

**Acceptance Criteria:**

- [ ] ...
- [ ] ...

---

### 4.3 {{Role C — e.g., Student}}

#### US-S1 — {{Capability}}

**As a** {{Role C}}, **I want** {{capability}} **so that** {{value}}.

**Acceptance Criteria:**

- [ ] ...
- [ ] ...

---

## 5. Out-of-Scope (Explicit)

The following will **not** be built in MVP:

- {{out-of-scope 1}}
- {{out-of-scope 2}}
- {{out-of-scope 3}}

---

## 6. Definition of Done (MVP)

The MVP is **done** when:

- [ ] All Phase 1 user stories above pass their acceptance criteria.
- [ ] CI runs lint + unit tests + build on every PR; pipeline is green on `main`.
- [ ] `docker-compose up` starts the system locally from a clean clone.
- [ ] Seed script populates baseline data.
- [ ] All declared events fire and are observable (log or in-memory listener).
- [ ] {{Public verification or demo path}} works end-to-end.

---
