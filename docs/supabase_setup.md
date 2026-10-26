# Cloud storage setup (Supabase)

Cloud storage is used by user scenario 10: sign-up, sign-in and synchronization
of sessions between devices. Without it the application works offline.

## 1. Project

1. Sign up at https://supabase.com and create a project (Free plan).
2. **Authentication → Sign In / Providers → Email**: turn off **Confirm email**,
   so that users can sign in right after registration.
3. **Project Settings → API Keys**: copy the **Project URL** and the
   **anon public** key (the **publishable** key also works).

## 2. Tables

Open **SQL Editor → New query**, paste and run:

```sql
create table public.sessions (
    id uuid primary key,
    user_id uuid not null default auth.uid() references auth.users(id) on delete cascade,
    started_at timestamp not null,
    finished_at timestamp,
    duration_sec integer,
    avg_engagement real,
    min_engagement real,
    source_type text,
    threshold integer
);

create table public.engagement_points (
    id uuid primary key,
    session_id uuid not null references public.sessions(id) on delete cascade,
    user_id uuid not null default auth.uid() references auth.users(id) on delete cascade,
    offset_sec double precision not null,
    value real not null
);

create index engagement_points_session_idx on public.engagement_points(session_id);

alter table public.sessions enable row level security;
alter table public.engagement_points enable row level security;

create policy "Users manage own sessions" on public.sessions
    for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy "Users manage own points" on public.engagement_points
    for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
```

If the tables were created with an earlier version of this guide, where
`offset_sec` was an integer, run once:

```sql
alter table public.engagement_points
    alter column offset_sec type double precision;
```

Estimates can come several times per second, so offsets have fractions.

Row Level Security guarantees that a user can read and change only their own
rows, even if someone obtains the public key.

## 3. Connecting the application

Create `~/.focuscore/cloud.json` (it is not stored in the repository):

```json
{"url": "https://<project>.supabase.co", "anon_key": "<anon public key>"}
```

Restart the application and open the «Профиль» section.

## What is stored on the server

Only numeric lecture data (time, duration, engagement scores) and the account
(email, first and last name). Video and audio are never sent.
