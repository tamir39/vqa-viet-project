# ✅ {{PROJECT_NAME}} — TASKS.md

> Execution checklist. Each task is sized for **1–3 hours** of focused work.
> Work top-to-bottom within a phase. Check off only when committed and verified.
> Conventions: `feat(<scope>): ...`, `fix(<scope>): ...` per repo CLAUDE.md.

---

## 🚀 Phase 1 — MVP Modular Monolith

### 1.1 Repository & Tooling Setup

- [ ] Initialize npm workspace at root (`server/`, `client/`, `shared/`).
- [ ] Add root `package.json` with workspaces and shared scripts (`lint`, `test`, `build`).
- [ ] Add root `tsconfig.base.json` and per-package `tsconfig.json` extending it.
- [ ] Add `.gitignore` covering `node_modules`, `dist`, `.env`, `coverage`.
- [ ] Add `.editorconfig` and `.prettierrc`.
- [ ] Add ESLint config (TypeScript + framework rules).
- [ ] Add Husky + lint-staged for pre-commit lint and format.

### 1.2 Shared Package (`shared/`)

- [ ] Scaffold `shared/` with `types/`, `constants/`, `index.ts`.
- [ ] Define cross-package types (User and any other shared entities).
- [ ] Define `DomainEvent<T>` type and `EventName` union.
- [ ] Define `event-names.ts` constants (canonical set per `PLANNING.md`).
- [ ] Define `roles.ts` constants.
- [ ] Wire shared package as a workspace dependency for client + server.

### 1.3 Server — Bootstrapping (`server/`)

- [ ] Generate framework skeleton ({{NestJS / Express}}).
- [ ] Create `core/config/` env loader with schema validation.
- [ ] Create `core/logger/` structured logger module.
- [ ] Create `core/errors/` with base error classes + normalizer.
- [ ] Create `core/database/` connection module.
- [ ] Create `core/events/` with internal `EventBus` service + `@OnEvent` decorator.
- [ ] Create global `error.middleware.ts` and register on app bootstrap.
- [ ] Create `auth.middleware.ts` and `role.middleware.ts` (RBAC guard).
- [ ] Add health endpoint `GET /health` returning DB + bus status.

### 1.4 Module: `auth`

- [ ] Scaffold module folder per `PLANNING.md` §3.3.
- [ ] Implement `POST /auth/register` (creates user with default role).
- [ ] Implement password hashing (bcrypt or argon2) in service layer.
- [ ] Implement `POST /auth/login` returning JWT.
- [ ] Add JWT strategy + `JwtAuthGuard`.
- [ ] Reject duplicate email with `409`; invalid credentials `401` (generic).
- [ ] Emit `user.registered` event on successful registration.
- [ ] Unit-test register + login happy path and 3 failure cases.

### 1.5 Module: `users`

- [ ] Scaffold module folder; implement model + repository.
- [ ] Implement `GET /users/me` (authenticated user profile).
- [ ] Implement `PATCH /users/me` (update profile fields).
- [ ] Implement admin-only list, role change, deactivate endpoints.
- [ ] Add audit log entry on every admin user action.
- [ ] Unit-test role guard rejects non-admin.

### 1.6 Module: `{{domain-1}}` (e.g., `courses`)

- [ ] Scaffold module; implement model + repository.
- [ ] Implement CRUD endpoints with role + ownership guards.
- [ ] Implement status lifecycle and valid-transition enforcement.
- [ ] Emit canonical events for state changes.
- [ ] Unit-test each service method.

### 1.7 Module: `{{domain-2}}`

- [ ] Scaffold module; implement model + repository.
- [ ] Implement endpoints per PRD acceptance criteria.
- [ ] Add guards as required.
- [ ] Emit canonical events.
- [ ] Unit-test each service method.

### 1.8 Module: `{{domain-3}}`

- [ ] Scaffold module; implement model + repository.
- [ ] Implement endpoints per PRD acceptance criteria.
- [ ] Add guards as required.
- [ ] Emit canonical events.
- [ ] Unit-test each service method.

### 1.9 Cross-Cutting (server)

- [ ] Add `class-validator` DTOs for every controller endpoint.
- [ ] Reject unknown fields globally (`whitelist: true, forbidNonWhitelisted: true`).
- [ ] Add API documentation tooling (Swagger / OpenAPI) at `/docs` (dev only).
- [ ] Add seed script `scripts/seed.ts` populating baseline data.
- [ ] Add `npm run seed` to root scripts.

### 1.10 Client — Bootstrapping (`client/`)

- [ ] Scaffold {{React / Next.js}} app.
- [ ] Configure TypeScript with `paths` to `shared/`.
- [ ] Add styling solution (Tailwind / CSS modules).
- [ ] Add React Query for data fetching.
- [ ] Add Zustand for light state (auth, current user).
- [ ] Add `services/` API client per server module.
- [ ] Add JWT interceptor (attach `Authorization`, handle 401 → logout).

### 1.11 Client — Core Pages

- [ ] Build `/register` page + form.
- [ ] Build `/login` page + form.
- [ ] Persist JWT (localStorage or httpOnly cookie).
- [ ] Build app shell (header, nav, route guards by role).
- [ ] Build feature pages per PRD user stories.

### 1.12 Events — End-to-End Wiring

- [ ] Verify all canonical events from `PLANNING.md` are emitted by their owners.
- [ ] Add an `events.dev-listener.ts` (dev-only) that logs every event for debugging.
- [ ] Document event contracts in `docs/events.md`.

### 1.13 Testing

- [ ] Set up Jest for server with module-by-module test folders.
- [ ] Write unit tests for each service's happy + error paths.
- [ ] Set up e2e tests covering the golden path defined in PRD.
- [ ] Set up Vitest (or Jest) for client.
- [ ] Write component tests for critical flows.
- [ ] Configure coverage threshold (≥70% lines on services).

### 1.14 Deployment (MVP)

- [ ] Write `docker/server.Dockerfile`.
- [ ] Write `docker/client.Dockerfile`.
- [ ] Write `docker-compose.yml` (db + server + client).
- [ ] Verify `docker-compose up` works from a clean clone.
- [ ] Add CI workflow: lint → typecheck → test → build.
- [ ] Add deploy step to the chosen hosts.
- [ ] Document local + prod setup in `README.md`.

### 1.15 MVP Definition-of-Done Verification

- [ ] All PRD user stories acceptance criteria checked off.
- [ ] CI green on `main`.
- [ ] Seed script runs in clean DB without errors.
- [ ] {{Public verification or demo path}} works end-to-end.
- [ ] Tag release `v1.0.0-mvp`.

---

## 🧱 Phase 2 — Microservices Preparation

### 2.1 Extract Boundaries

- [ ] Audit module imports — confirm zero cross-module logic imports.
- [ ] Document the public service interface of each module in `docs/module-contracts.md`.
- [ ] Identify candidates for first extraction.

### 2.2 Pluggable Event Bus

- [ ] Refactor `core/events/` to expose a `BusAdapter` interface (`publish`, `subscribe`).
- [ ] Implement `InMemoryBusAdapter` (current behavior).
- [ ] Implement `RedisPubSubBusAdapter` behind the same interface.
- [ ] Add `EVENT_BUS_DRIVER` env var to switch adapters.
- [ ] Add reconnection + retry logic for the Redis adapter.
- [ ] Add an event envelope with `id`, `name`, `occurredAt`, `producer`.

### 2.3 Configuration & Secrets

- [ ] Migrate config to per-service `.env` files in `services/<name>/`.
- [ ] Add a shared config package for cross-service constants.
- [ ] Document required secrets per service in `docs/operations.md`.

### 2.4 First Service Extraction (Pilot)

- [ ] Create `services/{{candidate}}/` as a standalone app.
- [ ] Move logic and DB tables to the service.
- [ ] Replace direct calls (none should exist) with event subscription.
- [ ] Decommission the module from the monolith.
- [ ] Update gateway/proxy or client to point to new service.
- [ ] Add e2e test that publishing an event triggers the service.

### 2.5 API Gateway / Edge

- [ ] Introduce a thin gateway that fronts monolith + extracted services.
- [ ] Centralize JWT verification at gateway.
- [ ] Add request tracing IDs propagated via headers + events.

### 2.6 Observability

- [ ] Add structured logs with correlation IDs across services.
- [ ] Add basic metrics (request count, error rate, event publish/consume).
- [ ] Add health endpoints (`/health`, `/ready`) per service.
- [ ] Wire to a free-tier observability backend.

### 2.7 Deployment & CI

- [ ] Update CI to build+test each service in parallel.
- [ ] Add per-service Docker images.
- [ ] Update `docker-compose.yml` to include extracted services.
- [ ] Add staging environment that runs full multi-service topology.

### 2.8 Phase-2 Definition-of-Done

- [ ] At least one service successfully extracted and consuming events.
- [ ] Monolith continues passing all Phase-1 acceptance tests.
- [ ] No domain module imports another module's internals.
- [ ] Bus driver is swappable via configuration without code change.
- [ ] Tag release `v2.0.0-services`.

---

## 📜 Rules

* Work sequentially within a phase; do not jump ahead.
* Commit only when a single TASKS bullet is complete and locally verified (per CLAUDE.md).
* Update this file immediately on completion (`- [x]`).
* Keep completed sections collapsed in `<details>` blocks once a sub-section is fully done.
