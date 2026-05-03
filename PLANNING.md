# 🧭 {{PROJECT_NAME}} — PLANNING.md

> Operational planning document. Pairs with `PRD.md` (the *what*) and optionally `ARCHITECTURE.md` (the *long-form design*).
> This file is the day-to-day map for engineers and AI agents working in this repo.

---

## 1. Architecture Summary

{{PROJECT_NAME}} is a **modular monolith** in **Phase 1**, with a **clear extraction path to microservices** in Phase 2+.

```
┌───────────────────────────────────┐
│   Client                          │
│   {{React / Next.js / other}}     │
└──────────────┬────────────────────┘
               │ REST (JSON, JWT)
               ▼
┌───────────────────────────────────┐
│   API Gateway ({{NestJS / Express}})│
│   Routing · Guards · Validation   │
└──────────────┬────────────────────┘
               │
   ┌───────────┼───────────┐
   ▼           ▼           ▼
┌──────┐  ┌──────┐   ┌──────────┐
│Mod A │  │Mod B │   │  Mod C   │  …all domain modules
└──┬───┘  └──┬───┘   └────┬─────┘
   │         │            │
   └─────────┼────────────┘
             ▼
   ┌─────────────────────────┐
   │   Service Layer         │  business logic
   └────────────┬────────────┘
                ▼
   ┌─────────────────────────┐
   │   Repository Layer      │  data access
   └────────────┬────────────┘
                ▼
   ┌─────────────────────────┐
   │   Database              │  {{Postgres / Mongo}}
   └─────────────────────────┘

           Internal Event Bus
   ─────────────────────────────────
   modules emit/listen to canonical events
```

**Core principles (binding):**

1. **Modular Domain Separation** — modules never cross-import logic. Cross-module communication is via events or via the public API of a service in the same module.
2. **Service-Oriented Core** — controllers thin, services own logic, repositories own DB.
3. **Derived State Preference** — computed values are derived, not stored.
4. **Event-Friendly Design** — every meaningful state change emits an event, even when no listener exists yet.
5. **Single Source of Truth per Concept** — schemas + types live in `shared/` and are imported by both client and server.

---

## 2. Module Catalog

| Module       | Responsibility                  | Owns Tables / Collections | Emits Events     | Listens To         |
| ------------ | ------------------------------- | ------------------------- | ---------------- | ------------------ |
| `auth`       | Authentication, token issuance  | —                         | `user.registered`| —                  |
| `users`      | Profile + admin user management | `users`                   | ...              | `user.registered`  |
| `{{domain1}}`| ...                             | `{{table}}`               | ...              | ...                |
| `{{domain2}}`| ...                             | `{{table}}`               | ...              | ...                |
| `events`     | In-process event bus            | —                         | (transport)      | (transport)        |

---

## 3. Folder Structure

### 3.1 Root

```
{{project_name}}/
├── client/                  # Frontend
├── server/                  # Backend
├── shared/                  # Shared types, constants, event contracts
├── docs/                    # Architecture + product docs
├── scripts/                 # Seed, migration helpers, one-off tools
├── docker/                  # Dockerfiles
├── docker-compose.yml
├── README.md
├── PRD.md
├── PLANNING.md
└── TASKS.md
```

### 3.2 Server (`server/`)

```
server/
├── src/
│   ├── modules/
│   │   ├── auth/
│   │   ├── users/
│   │   └── {{domain modules…}}
│   │
│   ├── core/
│   │   ├── database/
│   │   ├── config/
│   │   ├── logger/
│   │   ├── errors/
│   │   └── events/
│   │
│   ├── shared/
│   │   ├── utils/
│   │   ├── constants/
│   │   └── types/
│   │
│   ├── middleware/
│   │   ├── auth.middleware.ts
│   │   ├── role.middleware.ts
│   │   └── error.middleware.ts
│   │
│   ├── app.module.ts
│   └── main.ts
│
├── test/                    # e2e tests
├── tsconfig.json
└── package.json
```

### 3.3 Per-Module Layout (binding)

Every domain module under `server/src/modules/<name>/` follows this exact layout:

```
<module>/
├── <module>.module.ts        # Module declaration
├── <module>.controller.ts    # HTTP layer, validation, guards
├── <module>.service.ts       # Business logic
├── <module>.repository.ts    # DB access
├── <module>.model.ts         # Schema / entity
├── <module>.types.ts         # Module-local types
└── dto/                      # Request/response DTOs
```

### 3.4 Client (`client/`)

```
client/
├── src/
│   ├── pages/                # Routes
│   ├── components/           # Cross-feature reusable UI
│   ├── modules/              # Feature-based modules (mirror server)
│   ├── hooks/
│   ├── services/             # API clients (one per server module)
│   ├── store/                # Light state (Zustand / Redux)
│   ├── utils/
│   └── styles/
├── public/
├── tsconfig.json
└── package.json
```

### 3.5 Shared (`shared/`)

```
shared/
├── types/                    # Cross-package types
├── constants/
│   ├── roles.ts
│   └── event-names.ts
└── index.ts
```

> **Rule:** any type used by both `client/` and `server/` lives in `shared/`. Module-internal types stay in `<module>.types.ts`.

---

## 4. Data Model

> Conventions: `id` is a string (UUID or ObjectId). Timestamps `createdAt` / `updatedAt` exist on every entity but are omitted below for brevity.

### 4.1 User

```ts
User {
  id: string
  name: string
  email: string          // unique
  password: string       // hashed
  role: '{{role values}}'
  isActive: boolean
}
```

### 4.2 {{Entity B}}

```ts
{{EntityB}} {
  id: string
  // ...fields...
}
```

### 4.3 {{Entity C}}

```ts
{{EntityC}} {
  id: string
  // ...fields...
}
```

### 4.4 Event (in-memory contract — not persisted in MVP)

```ts
DomainEvent<T> {
  name: string           // e.g., '{{module}}.{{verb}}'
  payload: T
  occurredAt: Date
}
```

**Canonical event names** (from `shared/constants/event-names.ts`):

| Event                    | Payload                            | Emitter        |
| ------------------------ | ---------------------------------- | -------------- |
| `user.registered`        | `{ userId, role }`                 | `auth`         |
| `{{event-name}}`         | `{{payload shape}}`                | `{{module}}`   |

### 4.5 Relationship Map

```
User ─< {{relation}} >─ {{Entity}}
```

---

## 5. Security Model

| Concern                | Mechanism                                                                                       |
| ---------------------- | ----------------------------------------------------------------------------------------------- |
| **Authentication**     | JWT (HS256). Token in `Authorization: Bearer <jwt>`.                                            |
| **Password storage**   | `bcrypt` (cost ≥ 12) or `argon2id`. Never log or return passwords.                              |
| **Authorization**      | Role-based guards + ownership guards.                                                            |
| **Input validation**   | `class-validator` + `class-transformer` on every DTO. Reject unknown fields.                    |
| **Error normalization**| Single error middleware returns `{ code, message, details? }` with stable error codes.          |
| **Public endpoints**   | Document the explicit allow-list (e.g., `POST /auth/register`, `POST /auth/login`).            |
| **Rate limiting**      | Out of MVP scope (planned post-MVP via guard).                                                  |
| **Secrets**            | All via env vars. Never committed.                                                               |
| **CORS**               | Strict allow-list per environment.                                                               |

**RBAC matrix (template):**

|                          | {{Role A}} | {{Role B}} | {{Role C}} | Public |
| ------------------------ | :--------: | :--------: | :--------: | :----: |
| {{Action 1}}             |     ✅     |     ✅     |     ❌     |   ❌   |
| {{Action 2}}             |     ✅     |     ❌     |     ❌     |   ❌   |

---

## 6. Event System (Internal Event Bus)

### 6.1 Implementation (MVP)

- Built on **{{NestJS `EventEmitterModule`}}** (or a thin `EventBus` service wrapping Node's `EventEmitter`).
- **In-process only.** No transport, no persistence in MVP.
- Listeners register via decorators (`@OnEvent('event.name')`).

### 6.2 Why an event bus in a monolith

- Decouples domain logic between modules.
- Prepares the codebase for Phase 2 extraction.
- Enables analytics hooks later without touching domain code.

### 6.3 Phase-2 Migration Path

| Phase   | Bus Implementation                                              | Code change in domain modules     |
| ------- | --------------------------------------------------------------- | --------------------------------- |
| 1 (MVP) | In-process `EventEmitter`                                       | None (baseline)                   |
| 2       | Local bus + adapter to Redis Pub/Sub or NATS for selected events| None — only `events/` core changes|
| 3       | Full broker (Kafka/NATS) with persistence + replay              | None — only `events/` core changes|

> **Binding rule:** domain modules NEVER import the transport. They only `emit(name, payload)` and `@OnEvent(name)`.

---

## 7. Deployment

### 7.1 Local Development

- `docker-compose up` boots the full stack (DB, server, client).
- `npm run dev` for hot reload outside Docker.
- `scripts/seed.ts` populates baseline data.

### 7.2 CI Pipeline ({{GitHub Actions}})

```
PR opened / updated
   ↓
1. Install deps
2. Lint
3. Type check (tsc --noEmit)
4. Unit tests
5. Build server
6. Build client
7. Block merge if any step fails
```

### 7.3 Production Deployment

| Component | Host                    |
| --------- | ----------------------- |
| Frontend  | {{Vercel / static}}     |
| Backend   | {{Railway / Render / AWS}} |
| Database  | {{Managed Postgres}}    |
| Storage   | {{S3-compatible}}       |

### 7.4 Environments

- `local` — docker-compose, seeded.
- `staging` — auto-deploy from `develop`.
- `production` — auto-deploy from `main` after CI green.

### 7.5 Configuration

All environment-specific values via env vars; loaded + validated by `core/config/`.

```
DATABASE_URL=
JWT_SECRET=
JWT_EXPIRES_IN=1d
NODE_ENV=development|staging|production
PORT=3000
CORS_ORIGINS=
```

---

## 8. Roadmap

| Phase | Focus                          | Deliverable                                          |
| ----- | ------------------------------ | ---------------------------------------------------- |
| **1** | Modular monolith (MVP)         | All PRD §4 user stories pass acceptance criteria.    |
| **2** | Microservices preparation      | Extract first service; pluggable bus.                |
| **3** | Domain expansion               | {{future capabilities}}                              |

---

## 9. Decision Log

- **{{Framework}} chosen because** {{reason}}.
- **TypeScript everywhere** — shared types eliminate client/server contract drift.
- **In-process event bus first** — avoids infra cost while we validate domain boundaries.
- **Derived state** — prevents data-integrity bugs from denormalized truth.
- **No third-party APIs in MVP** — keeps the system runnable offline and on a free tier.

---

## Invariants (DO NOT VIOLATE)

* PRD.md is the source of truth for requirements.
* Do not add features not in PRD.md.
* Do not change high-level architecture without explicit approval.
* Do not delete code unless explicitly unsafe or deprecated.
* Modules do not cross-import logic.
* Computed values are derived, not stored.

---
