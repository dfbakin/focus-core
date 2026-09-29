# Настройка облачного хранилища (Supabase)

Облачное хранилище нужно для сценария 10: вход, регистрация и синхронизация
сессий между устройствами. Без него приложение работает в автономном режиме.

## 1. Проект

1. Зарегистрироваться на https://supabase.com и создать проект (бесплатный план).
2. **Authentication → Sign In / Providers → Email**: выключить **Confirm email**,
   чтобы вход работал сразу после регистрации.
3. **Project Settings → API**: скопировать **Project URL** и ключ **anon public**.

## 2. Таблицы

**SQL Editor → New query**, вставить и выполнить:

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
    offset_sec integer not null,
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

Row Level Security гарантирует, что пользователь видит и меняет только свои строки,
даже если кто-то узнает публичный ключ.

## 3. Подключение приложения

Создать файл `~/.focuscore/cloud.json` (в репозиторий не попадает):

```json
{"url": "https://<проект>.supabase.co", "anon_key": "<ключ anon public>"}
```

Перезапустить приложение и открыть раздел «Профиль».

## Что хранится на сервере

Только числовые данные занятий (время, длительность, оценки вовлечённости)
и учётная запись (адрес, имя, фамилия). Видео и звук не передаются.
