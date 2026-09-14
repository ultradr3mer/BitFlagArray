import type { Plugin } from "@opencode-ai/plugin"
import { readdir, stat } from "node:fs/promises"
import path from "node:path"

/**
 * Reflect-Watcher: every INTERVAL, if project files changed AND no chat is
 * running, prompts the plan agent to reflect on the changes. The prompt
 * includes the inactivity duration and asks for follow-up task suggestions.
 */
const INTERVAL_MS = 5 * 60 * 1000
const IGNORE = new Set([
  ".git", ".venv", "node_modules", "__pycache__", ".pytest_cache",
  ".opencode", ".mypy_cache", ".ruff_cache", "dist", "build",
])
const MAX_CHANGED_LISTED = 30

type Snapshot = Map<string, string> // relPath -> "mtimeMs:size"

export const ReflectWatcherPlugin: Plugin = async ({ client, worktree }) => {
  const root = worktree ?? "."
  let baseline: Snapshot | null = null
  let lastUserActivity = Date.now()
  let agentBusy = true // assume busy until we see a session.idle
  let reflecting = false

  await client.app.log({
    body: {
      service: "reflect-watcher",
      level: "info",
      message: `watching ${root} every ${INTERVAL_MS / 60000} min`,
    },
  })

  async function snapshot(dir: string, out: Snapshot = new Map(), rel = ""): Promise<Snapshot> {
    let entries
    try {
      entries = await readdir(dir, { withFileTypes: true })
    } catch {
      return out
    }
    for (const entry of entries) {
      if (IGNORE.has(entry.name)) continue
      const abs = path.join(dir, entry.name)
      const relPath = rel ? `${rel}/${entry.name}` : entry.name
      if (entry.isDirectory()) {
        await snapshot(abs, out, relPath)
      } else if (entry.isFile()) {
        try {
          const st = await stat(abs)
          out.set(relPath, `${st.mtimeMs}:${st.size}`)
        } catch {
          /* file vanished mid-scan */
        }
      }
    }
    return out
  }

  function diff(prev: Snapshot, next: Snapshot): { changed: string[]; deleted: string[] } {
    const changed: string[] = []
    const deleted: string[] = []
    for (const [p, sig] of next) {
      if (prev.get(p) !== sig) changed.push(p)
    }
    for (const p of prev.keys()) if (!next.has(p)) deleted.push(p)
    return { changed, deleted }
  }

  async function pickSessionId(): Promise<string | null> {
    try {
      const res = await client.session.list()
      const sessions = (res.data ?? res) as Array<Record<string, any>>
      const own = sessions.filter((s) => (s.directory ?? "").replace(/\\/g, "/") === String(root).replace(/\\/g, "/"))
      if (own.length > 0) {
        own.sort((a, b) => (b.time?.lastModified ?? 0) - (a.time?.lastModified ?? 0))
        return own[0].id
      }
      const created = await client.session.create({ body: { title: "reflect-watcher" } })
      const session = (created.data ?? created) as Record<string, any>
      return session?.id ?? null
    } catch {
      return null
    }
  }

  async function reflect(changed: string[], deleted: string[], inactiveMin: number) {
    reflecting = true
    try {
      const sessionId = await pickSessionId()
      if (!sessionId) return
      const lines = [
        "[reflect-watcher] Automatic reflection trigger after 5 minutes with file changes.",
        "",
        `The user has been inactive for ${inactiveMin} minute(s) — remind them of that.`,
        "",
        "Changed files since the last check:",
        ...changed.slice(0, MAX_CHANGED_LISTED).map((p) => `- ${p} (modified/added)`),
        ...deleted.slice(0, MAX_CHANGED_LISTED).map((p) => `- ${p} (deleted)`),
      ]
      if (changed.length + deleted.length > MAX_CHANGED_LISTED)
        lines.push(`- ... and ${changed.length + deleted.length - MAX_CHANGED_LISTED} more`)
      lines.push(
        "",
        "Tasks:",
        "1. Reflect on these changes: inspect them (e.g. git diff) and check for errors, inconsistencies or unfinished edits. Do NOT fix anything — you are in plan mode.",
        "2. Tell the user how long they have been inactive.",
        "3. Suggest 2-3 concrete follow-up tasks based on the changes.",
        "",
        "Keep the answer concise.",
      )
      const body: Record<string, any> = {
        agentID: "plan",
        parts: [{ type: "text", text: lines.join("\n") }],
      }
      try {
        await client.session.prompt({ path: { id: sessionId }, body })
      } catch {
        // agentID unsupported -> retry with default agent
        delete body.agentID
        await client.session.prompt({ path: { id: sessionId }, body })
      }
    } catch (e) {
      await client.app.log({
        body: { service: "reflect-watcher", level: "error", message: `reflect failed: ${String(e)}` },
      })
    } finally {
      reflecting = false
    }
  }

  async function check() {
    if (reflecting || agentBusy) return
    const next = await snapshot(root)
    if (baseline === null) {
      baseline = next
      return
    }
    const { changed, deleted } = diff(baseline, next)
    baseline = next
    if (changed.length === 0 && deleted.length === 0) return
    const inactiveMin = Math.max(1, Math.round((Date.now() - lastUserActivity) / 60000))
    try {
      await client.tui.showToast({
        body: {
          message: `reflect-watcher: ${changed.length + deleted.length} file(s) changed, inactive for ${inactiveMin} min — asking the plan agent to reflect`,
          variant: "info",
        },
      })
    } catch {
      /* no TUI attached */
    }
    await reflect(changed, deleted, inactiveMin)
  }

  const timer = setInterval(check, INTERVAL_MS)

  return {
    event: async ({ event }) => {
      const type = (event as any)?.type
      const props = (event as any)?.properties ?? {}
      if (type === "message.updated" || type === "message.part.updated") {
        if (props?.info?.role === "user") lastUserActivity = Date.now()
        else agentBusy = true // assistant streaming/working
      } else if (type === "session.idle" || type === "session.error") {
        agentBusy = false
      }
    },
  }
}
