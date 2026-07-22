import {
  createCliRenderer,
  BoxRenderable,
  TextRenderable,
  InputRenderable,
  InputRenderableEvents,
  ScrollBoxRenderable,
  MarkdownRenderable,
  ASCIIFontRenderable,
  SyntaxStyle,
  RGBA,
  t,
  fg,
  type CliRenderer,
} from "@opentui/core"
import { spawn, type ChildProcess } from "child_process"

// ─── Theme ───────────────────────────────────────────────────────────

const th = {
  bg: {
    root:         "#0a0a0c",
    panel:        "#141418",
    elevated:     "#1a1a1f",
    hover:        "#232329",
    border:       "#26262c",
    borderActive: "#3a3a44",
    tintUser:     "#0f1b14",
    tintReply:    "#171226",
    tintOrq:      "#1a1510",
  },
  fg: {
    primary:   "#e6e6e6",
    secondary: "#9aa0a6",
    muted:     "#5c5c66",
    faint:     "#3f3f47",
    accent:    "#f2a15d",
    accentDim: "#8a5a33",
    blue:      "#7aa2f7",
    green:     "#8fd98f",
    yellow:    "#e8c07d",
    orange:    "#f2a15d",
    red:       "#e0707a",
    purple:    "#c3a6f7",
    cyan:      "#67c0b5",
    white:     "#ffffff",
  },
  role: {
    user:    "#8fd98f",
    reply:   "#c3a6f7",
    orq:     "#f2a15d",
    info:    "#5c5c66",
    error:   "#e0707a",
    tool:    "#67c0b5",
    queue:   "#e8c07d",
  },
} as const

const mdStyle = SyntaxStyle.fromStyles({
  "markup.heading.1": { fg: RGBA.fromHex(th.fg.accent), bold: true },
  "markup.heading.2": { fg: RGBA.fromHex(th.fg.accent), bold: true },
  "markup.heading.3": { fg: RGBA.fromHex(th.fg.yellow) },
  "markup.list":      { fg: RGBA.fromHex(th.fg.accent) },
  "markup.raw":       { fg: RGBA.fromHex(th.fg.cyan) },
  "markup.bold":      { fg: RGBA.fromHex(th.fg.white), bold: true },
  "markup.italic":    { fg: RGBA.fromHex(th.fg.secondary), italic: true },
  "markup.quote":     { fg: RGBA.fromHex(th.fg.muted), italic: true },
  "markup.code":      { fg: RGBA.fromHex(th.fg.cyan) },
  "markup.link":      { fg: RGBA.fromHex(th.fg.blue), underline: true },
  "fenced_code":      { fg: RGBA.fromHex(th.fg.primary), bg: RGBA.fromHex(th.bg.panel) },
  "markup.table":     { fg: RGBA.fromHex(th.fg.secondary) },
  default:            { fg: RGBA.fromHex(th.fg.primary) },
})

// ─── Agent colors ────────────────────────────────────────────────────

const agentPalette = [
  "#67c0b5", "#7aa2f7", "#c3a6f7", "#8fd98f",
  "#e8c07d", "#e0707a", "#f2a15d", "#a8d8ea",
]

function hashStr(s: string): number {
  let h = 0
  for (let i = 0; i < s.length; i++) h = ((h << 5) - h + s.charCodeAt(i)) | 0
  return Math.abs(h)
}

function agentColor(name: string): string {
  const presets: Record<string, string> = { orchestrator: th.role.orq }
  if (presets[name]) return presets[name]
  return agentPalette[hashStr(name) % agentPalette.length]
}

function agentGlyph(name: string): string {
  return name === "orchestrator" ? "\u25CF" : "\u25C6"
}

const INTERNAL_TOOLS = new Set(["transfer_to_agent", "transfer_agent"])

// ─── Event types ─────────────────────────────────────────────────────

interface SkillInfo {
  id: string; name: string; description: string; tags: string[]; examples: string[]
}
interface AgentInfo {
  name: string; display_name: string; model: string; provider: string
  is_orchestrator: boolean
  skills: SkillInfo[]
  tools: string[]
}
interface InitEvent { type: "init"; agents: AgentInfo[] }
interface ResponseEvent { type: "response"; agent: string; text: string }
interface DelegateEvent { type: "delegate"; from_agent: string; to_agent: string }
interface ToolCallEvent { type: "tool_call"; agent: string; tool: string; args: Record<string, unknown> }
interface ToolResultEvent { type: "tool_result"; agent: string; tool: string; result: unknown }
interface ErrorEvent { type: "error"; text: string }
interface RetryEvent { type: "retry"; attempt: number; max_retries: number }
interface DoneEvent { type: "done" }
type Event = InitEvent | ResponseEvent | DelegateEvent | ToolCallEvent | ToolResultEvent | ErrorEvent | RetryEvent | DoneEvent

// ─── App ─────────────────────────────────────────────────────────────

export class A2ATuiApp {
  renderer!: CliRenderer

  // Layout
  headerBox!: BoxRenderable; headerLeft!: TextRenderable; headerRight!: TextRenderable
  scrollBox!: ScrollBoxRenderable; chatInput!: InputRenderable; inputRow!: BoxRenderable
  statusRow!: BoxRenderable; statusLeft!: TextRenderable; statusRight!: TextRenderable
  modeSeg!: BoxRenderable; modeSegText!: TextRenderable

  // Splash
  splashBox: BoxRenderable | null = null

  // Assistant card
  assistantBox: BoxRenderable | null = null
  assistantBody: BoxRenderable | null = null
  assistantSpinner: TextRenderable | null = null
  assistantHasReply = false
  thinkingRef: ReturnType<typeof setInterval> | null = null

  // Orchestrator thinking (before delegation)
  thinkingFold: BoxRenderable | null = null
  pendingReplyText = ""

  // Sub-agent reasoning (before tools)
  subAgentHasTools = false
  subAgentThinkingFold: BoxRenderable | null = null

  // Delegation marker
  delegateMarker: TextRenderable | null = null

  // Tool calls
  toolCallBoxes: BoxRenderable[] = []

  // Streaming markdown (reused for orchestrator pending reply AND sub-agent response)
  replyMd: MarkdownRenderable | null = null

  // Toast
  flashRef: ReturnType<typeof setTimeout> | null = null

  // State
  agents: AgentInfo[] = []
  streaming = false
  currentAgent = "orchestrator"
  hasDelegated = false
  lineBuf = ""
  hasStarted = false
  msgCounter = 0
  replyCounter = 0
  proc: ChildProcess | null = null
  palette: CommandPalette | null = null

  // ── Helpers ──────────────────────────────────────────────────────

  private nowHM(): string {
    const d = new Date()
    return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`
  }

  private msgMaxWidth(): number {
    return Math.max(24, Math.min(92, (this.renderer?.width ?? 90) - 6))
  }

  private readonly spinnerFrames =
    ["\u280B", "\u2819", "\u2839", "\u2838", "\u283C", "\u2834", "\u2826", "\u2827", "\u2807", "\u280F"]

  private setMode(label: string, colorHex: string): void {
    if (this.modeSegText) this.modeSegText.content = label
    if (this.modeSeg) this.modeSeg.backgroundColor = colorHex
  }

  private flashStatus(msg: string, colorHex: string, ms = 1600): void {
    if (this.flashRef) clearTimeout(this.flashRef)
    this.statusLeft.content = msg
    this.statusLeft.fg = colorHex
    this.flashRef = setTimeout(() => { this.flashRef = null; this.updateStatus() }, ms)
  }

  private showToast(msg: string): void {
    this.flashStatus(`\u2713 ${msg}`, th.fg.green, 2500)
  }

  // ── Build UI ─────────────────────────────────────────────────────

  async build(r: CliRenderer): Promise<void> {
    this.renderer = r
    r.setBackgroundColor(th.bg.root)
    r.setCursorColor(RGBA.fromHex(th.fg.accent))
    const root = r.root
    root.width = "100%"; root.height = "100%"

    // Header
    this.headerBox = new BoxRenderable(r, {
      id: "header", width: "100%", height: 1,
      backgroundColor: th.bg.root, flexDirection: "row",
      paddingLeft: 2, paddingRight: 2,
    })
    this.headerLeft = new TextRenderable(r, { id: "header-left", content: "\u25C6 a2a", fg: th.fg.accent, flexGrow: 1 })
    this.headerRight = new TextRenderable(r, { id: "header-right", content: "starting\u2026", fg: th.fg.muted })
    this.headerBox.add(this.headerLeft)
    this.headerBox.add(this.headerRight)
    root.add(this.headerBox)

    const divider = new BoxRenderable(r, {
      id: "header-divider", width: "100%", height: 1,
      backgroundColor: th.bg.root, borderStyle: "single",
      borderColor: th.bg.border, border: ["bottom"],
    })
    root.add(divider)

    // ScrollBox
    this.scrollBox = new ScrollBoxRenderable(r, {
      id: "chat-scroll", flexGrow: 1, width: "100%",
      stickyScroll: true, stickyStart: "bottom",
      paddingTop: 1, paddingBottom: 1,
      scrollbarOptions: { trackOptions: { foregroundColor: th.bg.borderActive, backgroundColor: th.bg.root } },
      rootOptions: { backgroundColor: th.bg.root },
      wrapperOptions: { backgroundColor: th.bg.root },
      viewportOptions: { backgroundColor: th.bg.root },
      contentOptions: { backgroundColor: th.bg.root },
    })
    root.add(this.scrollBox)

    // Input row
    this.inputRow = new BoxRenderable(r, {
      id: "input-row", width: "100%", height: 3,
      backgroundColor: th.bg.panel, borderStyle: "rounded", border: true,
      borderColor: th.bg.borderActive, flexDirection: "row",
      marginLeft: 1, marginRight: 1, paddingLeft: 1, paddingRight: 1,
    })
    const promptGlyph = new TextRenderable(r, { content: "\u203A ", fg: th.fg.accent })
    this.chatInput = new InputRenderable(r, {
      id: "chat-input", flexGrow: 1,
      backgroundColor: th.bg.panel, focusedBackgroundColor: th.bg.panel,
      placeholder: "ask the orchestrator\u2026   /help   \u00B7   ^P palette",
      placeholderColor: th.fg.muted, textColor: th.fg.primary, cursorColor: th.fg.accent,
    })
    this.chatInput.on(InputRenderableEvents.ENTER, (v: string) => this.onSubmit(v))
    this.inputRow.add(promptGlyph)
    this.inputRow.add(this.chatInput)
    root.add(this.inputRow)

    // Status bar
    this.statusRow = new BoxRenderable(r, {
      id: "status-bar", width: "100%", height: 1,
      backgroundColor: th.bg.root, flexDirection: "row",
      paddingLeft: 1, paddingRight: 2,
    })
    this.modeSeg = new BoxRenderable(r, {
      id: "mode-seg", height: 1,
      backgroundColor: th.fg.accent, paddingLeft: 1, paddingRight: 1, marginRight: 1,
    })
    this.modeSegText = new TextRenderable(r, { content: "boot", fg: th.bg.root })
    this.modeSeg.add(this.modeSegText)
    this.statusLeft = new TextRenderable(r, { content: "\u25CF starting\u2026", fg: th.fg.muted, flexGrow: 1 })
    this.statusRight = new TextRenderable(r, { content: "^P palette \u00B7 ^L clear \u00B7 /help", fg: th.fg.faint })
    this.statusRow.add(this.modeSeg)
    this.statusRow.add(this.statusLeft)
    this.statusRow.add(this.statusRight)
    root.add(this.statusRow)

    this.palette = new CommandPalette(r, this)
    r.keyInput.on("keypress", (key: any) => this.onGlobalKey(key))

    r.on("selection", (sel: any) => {
      if (sel?.isActive) {
        const text: string = sel.getSelectedText?.() ?? ""
        if (text) {
          r.copyToClipboardOSC52(text)
          try {
            const xclip = spawn("xclip", ["-selection", "clipboard"], { stdio: "pipe" })
            xclip.stdin?.write(text); xclip.stdin?.end()
          } catch {}
          this.showToast("copied to clipboard")
        }
      }
    })

    this.showSplash()
    this.chatInput.focus()
  }

  // ── Splash ──────────────────────────────────────────────────────

  showSplash(): void {
    this.splashBox = new BoxRenderable(this.renderer, {
      id: "splash", width: "100%", flexGrow: 1,
      justifyContent: "center", alignItems: "center", flexDirection: "column", gap: 1,
    })
    let logo: TextRenderable | ASCIIFontRenderable
    try {
      logo = new ASCIIFontRenderable(this.renderer, {
        id: "splash-logo", text: "A2A", font: "tiny", color: RGBA.fromHex(th.fg.accent),
      })
    } catch {
      logo = new TextRenderable(this.renderer, { content: "A2A", fg: th.fg.accent })
    }
    const sub = new TextRenderable(this.renderer, {
      content: "agent-to-agent   \u00B7   google ADK   \u00B7   multi-agent orchestrator",
      fg: th.fg.secondary,
    })
    const hint = new TextRenderable(this.renderer, {
      content: "type your query below     \u00B7     /help for commands     \u00B7     ^P palette",
      fg: th.fg.muted,
    })
    this.splashBox.add(logo as any)
    this.splashBox.add(sub)
    this.splashBox.add(hint)
    this.scrollBox.add(this.splashBox)
  }

  hideSplash(): void {
    if (this.splashBox) {
      this.scrollBox.remove(this.splashBox)
      this.splashBox.destroyRecursively()
      this.splashBox = null
    }
  }

  // ── Event stream ──────────────────────────────────────────────────

  boot(projectDir: string): void {
    this.proc = spawn("python3", ["orchestrator.py", "--json-events"], {
      cwd: projectDir, stdio: ["pipe", "pipe", "pipe"],
    })
    const stdout = this.proc.stdout
    if (!stdout) return
    stdout.setEncoding("utf8")
    stdout.on("data", (chunk: string) => {
      this.lineBuf += chunk
      const lines = this.lineBuf.split("\n")
      this.lineBuf = lines.pop() ?? ""
      for (const line of lines) {
        if (!line.trim()) continue
        try { this.handleEvent(JSON.parse(line) as Event) } catch {}
      }
    })
    const stderr = this.proc.stderr
    if (stderr) {
      stderr.setEncoding("utf8")
      stderr.on("data", (chunk: string) => {
        for (const line of chunk.split("\n").filter(Boolean)) console.error("[py]", line)
      })
    }
    this.proc.on("exit", (code: number | null) => {
      if (!this.hasStarted && code !== 0) this.addError(`Backend exited with code ${code}`)
      this.setMode("dead", th.fg.red)
      this.flashStatus("\u2717 backend disconnected", th.fg.red)
    })
    this.setMode("boot", th.fg.yellow)
    this.updateStatus("\u25CF booting orchestrator\u2026")
  }

  killBackend(): void {
    if (this.proc) {
      try { this.proc.stdin?.end(); this.proc.kill("SIGTERM") } catch {}
      this.proc = null
    }
  }

  handleEvent(ev: Event): void {
    switch (ev.type) {
      case "init":
        this.agents = ev.agents
        this.refreshHeader(); this.updateStatus(); this.setMode("chat", th.fg.accent)
        if (this.splashBox) this.refreshSplashWithAgents(ev.agents)
        break

      case "response":
        this.handleResponse(ev.agent, ev.text)
        break

      case "delegate":
        this.hasDelegated = true
        this.currentAgent = ev.to_agent
        this.convertReplyToThinking()
        this.showDelegationMarker(ev.from_agent, ev.to_agent)
        break

      case "tool_call":
        if (INTERNAL_TOOLS.has(ev.tool)) break
        this.handleToolCall(ev.agent, ev.tool, ev.args)
        break

      case "tool_result":
        if (INTERNAL_TOOLS.has(ev.tool)) break
        this.showToolResult(ev.tool, ev.result)
        break

      case "error":
        this.addError(ev.text)
        break

      case "retry":
        this.cleanupCurrentCard()
        this.flashStatus(`retrying (${ev.attempt}/${ev.max_retries})\u2026`, th.fg.yellow)
        break

      case "done":
        this.finishTurn()
        break
    }
  }

  // ── Response handling ────────────────────────────────────────────

  private handleResponse(agent: string, text: string): void {
    if (agent === "orchestrator" && !this.hasDelegated) {
      this.addPendingReply(text)
    } else if (agent !== "orchestrator" && this.hasDelegated) {
      this.addSubAgentText(agent, text)
    } else {
      this.addFinalResponse(text)
    }
  }

  // ── User messages ────────────────────────────────────────────────

  addUserMsg(text: string): void {
    this.msgCounter++
    const hm = this.nowHM()
    const container = new BoxRenderable(this.renderer, {
      width: "100%", flexDirection: "column", marginTop: 2,
      paddingLeft: 2, paddingRight: 2, alignItems: "flex-end",
    })
    const bubble = new BoxRenderable(this.renderer, {
      id: `msg-user-${this.msgCounter}`, flexDirection: "column",
      maxWidth: this.msgMaxWidth(),
      backgroundColor: th.bg.tintUser, borderStyle: "rounded", border: true,
      borderColor: th.role.user, paddingLeft: 1, paddingRight: 1,
    })
    bubble.add(new TextRenderable(this.renderer, {
      content: t`${fg(th.role.user)("you")} ${fg(th.fg.muted)("\u00B7 " + hm)}`,
    }))
    bubble.add(new TextRenderable(this.renderer, {
      content: text, fg: th.fg.primary, selectable: true,
    }))
    container.add(bubble)
    this.scrollBox.add(container)
  }

  // ── Assistant card ──────────────────────────────────────────────

  private ensureCard(): void {
    if (this.assistantBody) return
    this.assistantHasReply = false
    this.toolCallBoxes = []
    this.replyMd = null
    this.hasDelegated = false
    this.pendingReplyText = ""
    this.thinkingFold = null
    this.subAgentHasTools = false
    this.subAgentThinkingFold = null
    this.delegateMarker = null

    this.replyCounter++
    this.assistantBox = new BoxRenderable(this.renderer, {
      id: `assistant-${this.replyCounter}`, flexDirection: "column",
      marginTop: 1, marginLeft: 2, marginRight: 2, maxWidth: this.msgMaxWidth(),
      alignSelf: "flex-start",
      backgroundColor: th.bg.tintOrq, borderStyle: "rounded", border: true,
      borderColor: th.role.orq, paddingLeft: 1, paddingRight: 1,
    })
    const hm = this.nowHM()
    this.assistantBox.add(new TextRenderable(this.renderer, {
      content: t`${fg(th.role.orq)("\u25CF orchestrator")} ${fg(th.fg.muted)("\u00B7 " + hm)}`,
    }))
    this.assistantBody = new BoxRenderable(this.renderer, { flexDirection: "column", paddingTop: 1 })
    this.assistantBox.add(this.assistantBody)

    this.assistantSpinner = new TextRenderable(this.renderer, {
      content: `${this.spinnerFrames[0]} thinking\u2026`, fg: th.fg.muted,
    })
    this.assistantBody.add(this.assistantSpinner)
    this.startSpinner()
    this.scrollBox.add(this.assistantBox)
  }

  private startSpinner(): void {
    let i = 0
    this.thinkingRef = setInterval(() => {
      if (!this.assistantSpinner) {
        if (this.thinkingRef) clearInterval(this.thinkingRef)
        this.thinkingRef = null
        return
      }
      i = (i + 1) % this.spinnerFrames.length
      this.assistantSpinner.content = `${this.spinnerFrames[i]} thinking\u2026`
    }, 90)
  }

  private stopSpinner(): void {
    if (this.thinkingRef) { clearInterval(this.thinkingRef); this.thinkingRef = null }
    if (this.assistantSpinner) {
      this.assistantBody?.remove(this.assistantSpinner)
      this.assistantSpinner.destroy()
      this.assistantSpinner = null
    }
  }

  private pinSpinnerLast(): void {
    if (this.assistantSpinner && this.assistantBody) {
      this.assistantBody.remove(this.assistantSpinner)
      this.assistantBody.add(this.assistantSpinner)
    }
  }

  // ── Pending reply (orchestrator text before delegation) ───────────

  private addPendingReply(text: string): void {
    this.ensureCard()
    this.pendingReplyText += text
    this.stopSpinner()
    this.assistantHasReply = true

    if (!this.replyMd) {
      this.replyMd = new MarkdownRenderable(this.renderer, {
        content: text, syntaxStyle: mdStyle, streaming: true, conceal: true,
      })
      this.assistantBody!.add(this.replyMd)
    } else {
      this.replyMd.content = (this.replyMd.content ?? "") + text
    }
  }

  private convertReplyToThinking(): void {
    if (!this.pendingReplyText || !this.assistantBody) return

    if (this.replyMd) {
      this.assistantBody.remove(this.replyMd)
      this.replyMd.destroyRecursively()
      this.replyMd = null
    }

    const lineCount = this.pendingReplyText.split("\n").length
    this.thinkingFold = new BoxRenderable(this.renderer, {
      flexDirection: "column", marginBottom: 1,
      border: true, borderStyle: "rounded", borderColor: th.bg.border,
      paddingLeft: 1, paddingRight: 1,
    })
    this.thinkingFold.add(new TextRenderable(this.renderer, {
      content: t`${fg(th.fg.accentDim)("\u25BE reasoning")} ${fg(th.fg.faint)(lineCount + " lines")}`,
    }))
    this.thinkingFold.add(new TextRenderable(this.renderer, {
      content: this.pendingReplyText, fg: th.fg.faint,
      selectable: true, maxWidth: this.msgMaxWidth() - 6,
    }))
    this.assistantBody.add(this.thinkingFold, 0)
    this.pendingReplyText = ""
    this.assistantHasReply = false
    this.startSpinner()
  }

  // ── Sub-agent text (reasoning before tools, or response after) ───

  private addSubAgentText(agent: string, text: string): void {
    this.ensureCard()

    if (!this.subAgentHasTools) {
      // Could be reasoning or response — show as streaming markdown
      this.stopSpinner()
      if (!this.replyMd) {
        this.replyMd = new MarkdownRenderable(this.renderer, {
          content: text, syntaxStyle: mdStyle, streaming: true, conceal: true,
        })
        this.assistantBody!.add(this.replyMd)
        this.pinSpinnerLast()
      } else {
        this.replyMd.content = (this.replyMd.content ?? "") + text
      }
      this.assistantHasReply = true
    } else {
      // After tools — this is the actual response
      this.stopSpinner()
      this.assistantHasReply = true
      if (!this.replyMd) {
        // Subtle separator before response (not ▼ since more tools may follow)
        this.replyMd = new MarkdownRenderable(this.renderer, {
          content: text, syntaxStyle: mdStyle, streaming: true, conceal: true,
          marginTop: 1,
        })
        this.assistantBody!.add(this.replyMd)
      } else {
        this.replyMd.content = (this.replyMd.content ?? "") + text
      }
    }
    this.updateCardAgent(agent)
  }

  private convertSubAgentTextToThinking(): void {
    if (!this.replyMd || !this.assistantBody) return
    const buf = this.replyMd.content ?? ""
    if (!buf.trim()) return

    this.assistantBody.remove(this.replyMd)
    this.replyMd.destroyRecursively()
    this.replyMd = null

    const lineCount = buf.split("\n").length
    const fold = new BoxRenderable(this.renderer, {
      flexDirection: "column", marginBottom: 1,
      border: true, borderStyle: "rounded", borderColor: th.bg.border,
      paddingLeft: 1, paddingRight: 1,
    })
    fold.add(new TextRenderable(this.renderer, {
      content: t`${fg(th.fg.accentDim)("\u25BE reasoning")} ${fg(th.fg.faint)(lineCount + " lines")}`,
    }))
    fold.add(new TextRenderable(this.renderer, {
      content: buf, fg: th.fg.faint, selectable: true, maxWidth: this.msgMaxWidth() - 6,
    }))
    this.assistantBody.add(fold)
    this.subAgentThinkingFold = fold
    this.assistantHasReply = false
    this.startSpinner()
  }

  // ── Delegation marker ────────────────────────────────────────────

  private showDelegationMarker(fromAgent: string, toAgent: string): void {
    this.ensureCard()
    const fromColor = agentColor(fromAgent)
    const toColor = agentColor(toAgent)

    // Vertical connector: │ then ▼ with agent names
    this.assistantBody!.add(new TextRenderable(this.renderer, {
      content: t`  ${fg(th.fg.faint)("\u2502")}`,
      fg: th.fg.faint,
    }))
    this.delegateMarker = new TextRenderable(this.renderer, {
      content: t`  ${fg(RGBA.fromHex(toColor))("\u25BC")} ${fg(RGBA.fromHex(fromColor))(agentGlyph(fromAgent) + " " + fromAgent)} ${fg(th.fg.muted)("\u2192")} ${fg(RGBA.fromHex(toColor))(agentGlyph(toAgent) + " " + toAgent)}`,
      fg: th.fg.secondary,
      marginBottom: 0,
    })
    this.assistantBody!.add(this.delegateMarker)
    this.pinSpinnerLast()
  }

  // ── Tool calls ────────────────────────────────────────────────────

  private handleToolCall(agent: string, tool: string, args: Record<string, unknown>): void {
    this.ensureCard()

    // If sub-agent has text but no tools yet, convert text to thinking
    if (agent !== "orchestrator" && !this.subAgentHasTools && this.replyMd) {
      this.convertSubAgentTextToThinking()
    }

    // Finalize any existing response text (more tools coming)
    const hadReply = this.replyMd !== null
    if (this.replyMd) {
      this.replyMd.streaming = false
      this.replyMd = null
    }

    // Add single │ separator before tool (if not the first after delegation)
    if (this.subAgentHasTools || hadReply) {
      this.assistantBody!.add(new TextRenderable(this.renderer, {
        content: t`  ${fg(th.fg.faint)("\u2502")}`,
        fg: th.fg.faint,
      }))
      this.pinSpinnerLast()
    }

    this.subAgentHasTools = true
    this.stopSpinner()

    const argsStr = JSON.stringify(args, null, 1)
    let argsLine = ""
    if (argsStr && argsStr !== "{}") {
      argsLine = argsStr.replace(/\n\s*/g, " ").slice(0, 68)
      if (argsStr.length > 68) argsLine += "\u2026"
    }

    const box = new BoxRenderable(this.renderer, { flexDirection: "column", marginTop: 0 })
    box.add(new TextRenderable(this.renderer, {
      content: t`  ${fg(th.fg.faint)("\u251C\u2500")} ${fg(th.role.tool)("\u23FA " + tool)}`,
    }))
    if (argsLine) {
      box.add(new TextRenderable(this.renderer, {
        content: t`  ${fg(th.fg.faint)("\u2502")}   ${fg(th.fg.secondary)("\u23BF " + argsLine)}`,
        selectable: true,
      }))
    }
    ;(box as any).__tool = tool
    this.assistantBody!.add(box)
    this.pinSpinnerLast()
    this.toolCallBoxes.push(box)
    this.updateCardAgent(agent)
  }

  showToolResult(tool: string, result: unknown): void {
    this.ensureCard()
    let preview = ""
    if (typeof result === "object" && result !== null) {
      const r = result as Record<string, unknown>
      const content = r.content as Array<Record<string, unknown>> | undefined
      if (content && content[0]?.text) {
        preview = String(content[0].text).slice(0, 100).replace(/\n.*$/s, "")
      } else {
        preview = JSON.stringify(result).slice(0, 80)
      }
    } else {
      preview = String(result).slice(0, 80)
    }

    for (let i = this.toolCallBoxes.length - 1; i >= 0; i--) {
      const box = this.toolCallBoxes[i]
      if ((box as any).__tool === tool) {
        const label = box.getChildren()[0] as TextRenderable | undefined
        if (label) label.content = t`  ${fg(th.fg.faint)("\u251C\u2500")} ${fg(th.fg.green)("\u2714 " + tool)}`
        if (preview) {
          const short = preview.length > 80 ? preview.slice(0, 77) + "\u2026" : preview
          box.add(new TextRenderable(this.renderer, {
            content: t`  ${fg(th.fg.faint)("\u2502")}   ${fg(th.fg.muted)(short)}`,
            selectable: true,
          }))
        }
        break
      }
    }
  }

  private updateCardAgent(name: string): void {
    if (!this.assistantBox) return
    const color = agentColor(name)
    const hm = this.nowHM()
    const glyph = agentGlyph(name)
    const kids = this.assistantBox.getChildren()
    if (kids.length > 0 && kids[0] instanceof TextRenderable) {
      ;(kids[0] as TextRenderable).content =
        t`${fg(RGBA.fromHex(color))(glyph + " " + name)} ${fg(th.fg.muted)("\u00B7 " + hm)}`
    }
    if (name !== "orchestrator") {
      this.assistantBox.borderColor = RGBA.fromHex(color)
      this.assistantBox.backgroundColor = th.bg.tintReply
    }
  }

  // ── Final orchestrator response ─────────────────────────────────

  private addFinalResponse(text: string): void {
    this.ensureCard()
    this.stopSpinner()
    this.assistantHasReply = true
    this.updateCardAgent("orchestrator")

    if (!this.replyMd) {
      this.replyMd = new MarkdownRenderable(this.renderer, {
        content: text, syntaxStyle: mdStyle, streaming: true, conceal: true,
        marginTop: 1,
      })
      this.assistantBody!.add(this.replyMd)
    } else {
      this.replyMd.content = (this.replyMd.content ?? "") + text
    }
  }

  // ── Error ────────────────────────────────────────────────────────

  addError(text: string): void {
    if (this.assistantBox && !this.assistantHasReply && this.toolCallBoxes.length === 0) {
      this.scrollBox.remove(this.assistantBox)
      this.assistantBox.destroyRecursively()
      this.assistantBox = null
      this.assistantBody = null
    }
    this.stopSpinner()

    const container = new BoxRenderable(this.renderer, {
      width: "100%", flexDirection: "column", marginTop: 1,
      paddingLeft: 2, paddingRight: 2,
    })
    container.add(new TextRenderable(this.renderer, {
      content: t`${fg(th.role.error)("\u2717 error")} ${fg(th.fg.muted)("\u00B7 " + this.nowHM())}`,
    }))
    container.add(new TextRenderable(this.renderer, {
      content: text, fg: th.fg.secondary, selectable: true, paddingLeft: 2,
    }))
    this.scrollBox.add(container)
  }

  // ── Finish turn ──────────────────────────────────────────────────

  private cleanupCurrentCard(): void {
    this.stopSpinner()
    if (this.assistantBox) {
      this.scrollBox.remove(this.assistantBox)
      this.assistantBox.destroyRecursively()
    }
    this.assistantBox = null
    this.assistantBody = null
    this.replyMd = null
    this.toolCallBoxes = []
    this.thinkingFold = null
    this.subAgentThinkingFold = null
    this.subAgentHasTools = false
    this.delegateMarker = null
    this.pendingReplyText = ""
    this.hasDelegated = false
    this.currentAgent = "orchestrator"
  }

  finishTurn(): void {
    this.stopSpinner()
    if (this.replyMd) this.replyMd.streaming = false
    if (!this.assistantHasReply && this.toolCallBoxes.length === 0 && this.assistantBox) {
      this.scrollBox.remove(this.assistantBox)
      this.assistantBox.destroyRecursively()
    }
    this.assistantBox = null
    this.assistantBody = null
    this.replyMd = null
    this.toolCallBoxes = []
    this.thinkingFold = null
    this.subAgentThinkingFold = null
    this.subAgentHasTools = false
    this.delegateMarker = null
    this.pendingReplyText = ""
    this.hasDelegated = false
    this.streaming = false
    this.currentAgent = "orchestrator"
    this.setMode("chat", th.fg.accent)
    this.updateStatus()
  }

  // ── Header / Status ─────────────────────────────────────────────

  private refreshHeader(): void {
    if (this.agents.length === 0) return
    const subAgents = this.agents.filter(a => !a.is_orchestrator)
    this.headerRight.content = `${subAgents.length} agents \u00B7 ${subAgents.map(a => a.name).join(", ")}`
    this.headerRight.fg = th.fg.secondary
  }

  updateStatus(msg?: string): void {
    if (msg) {
      this.statusLeft.content = msg
      this.statusLeft.fg = th.fg.secondary
    } else if (this.agents.length > 0) {
      const orqModel = this.agents.find(a => a.is_orchestrator)?.model ?? "?"
      this.statusLeft.content = `\u25CF ${orqModel}`
      this.statusLeft.fg = th.fg.muted
    } else {
      this.statusLeft.content = "\u25CB starting\u2026"
      this.statusLeft.fg = th.fg.muted
    }
  }

  private refreshSplashWithAgents(agents: AgentInfo[]): void {
    if (!this.splashBox) return
    const subAgents = agents.filter(a => !a.is_orchestrator)
    if (subAgents.length === 0) return

    const existing = this.splashBox.getChildren()
    for (const child of existing) {
      if (child instanceof TextRenderable && (child as any).id?.startsWith("splash-agents")) {
        this.splashBox.remove(child)
        child.destroy()
      }
    }

    const lines = subAgents.map(a => {
      const skillCount = a.skills?.length ?? 0
      const toolCount = a.tools?.length ?? 0
      return `${agentGlyph(a.name)} ${a.name} (${a.model}) ${skillCount} skills${toolCount > 0 ? ", " + toolCount + " tools" : ""}`
    })

    const agentsText = new TextRenderable(this.renderer, {
      content: lines.join("\n"), fg: th.fg.secondary,
    })
    ;(agentsText as any).id = "splash-agents"
    this.splashBox.add(agentsText)
  }

  // ── Clear ────────────────────────────────────────────────────────

  softClear(): void {
    this.hideSplash()
    this.stopSpinner()
    if (this.assistantBox) {
      this.scrollBox.remove(this.assistantBox)
      this.assistantBox.destroyRecursively()
    }
    this.assistantBox = null
    this.assistantBody = null
    this.replyMd = null
    this.toolCallBoxes = []
    this.thinkingFold = null
    this.subAgentThinkingFold = null
    this.subAgentHasTools = false
    this.delegateMarker = null
    for (const child of this.scrollBox.getChildren()) {
      this.scrollBox.remove(child)
      child.destroyRecursively()
    }
    this.msgCounter = 0
    this.replyCounter = 0
    this.showSplash()
    this.refreshSplashWithAgents(this.agents)
    this.chatInput.focus()
  }

  // ── Slash commands ──────────────────────────────────────────────

  async handleSlash(cmd: string): Promise<void> {
    this.hideSplash()
    const parts = cmd.trim().split(/\s+/)
    const base = parts[0]!.toLowerCase()

    switch (base) {
      case "/help":
        this.addInfo(
          "Commands\n" +
          "  /agents      List connected agents and their skills\n" +
          "  /clear       Clear conversation\n" +
          "  /help        Show this help\n" +
          "  /exit        Quit\n\n" +
          "Shortcuts\n" +
          "  ^P    Command palette\n" +
          "  ^L    Clear conversation\n" +
          "  ^Y    Copy selected text\n" +
          "  ^C    Quit\n" +
          "  esc   Cancel / blur input"
        )
        break

      case "/agents": {
        if (this.agents.length === 0) { this.addInfo("No agents loaded yet."); break }
        const lines: string[] = ["Connected Agents"]
        for (const a of this.agents) {
          const role = a.is_orchestrator ? "orchestrator" : "sub-agent"
          lines.push(`${agentGlyph(a.name)} ${a.display_name} (${role})`)
          lines.push(`    model: ${a.model}  \u00B7  provider: ${a.provider}`)
          if (a.skills?.length) {
            lines.push("    skills:")
            for (const s of a.skills) lines.push(`      - ${s.id}: ${s.description.trim()}`)
          }
          if (a.tools?.length) lines.push(`    tools: ${a.tools.join(", ")}`)
          lines.push("")
        }
        this.addInfo(lines.join("\n"))
        break
      }

      case "/clear": this.softClear(); break
      case "/exit": this.killBackend(); this.renderer.destroy(); process.exit(0); break
      default: this.addInfo(`Unknown command: ${base}  \u00B7  /help for available commands`)
    }
  }

  addInfo(text: string): void {
    const container = new BoxRenderable(this.renderer, {
      width: "100%", flexDirection: "column", marginTop: 1, paddingLeft: 2, paddingRight: 2,
    })
    container.add(new TextRenderable(this.renderer, { content: "\u203B info", fg: th.fg.muted }))
    container.add(new TextRenderable(this.renderer, {
      content: text, fg: th.fg.secondary, selectable: true, paddingLeft: 2,
    }))
    this.scrollBox.add(container)
  }

  // ── Global keyboard ──────────────────────────────────────────────

  onGlobalKey(key: any): void {
    if (this.palette?.visible) {
      if (key.name === "escape") { this.palette.hide(); return }
      if (key.name === "up" || key.name === "k") { this.palette.navigate(-1); return }
      if (key.name === "down" || key.name === "j") { this.palette.navigate(1); return }
      if (key.name === "enter" || key.name === "return") { this.palette.execute(); return }
      return
    }
    if (key.ctrl && key.name === "p") { this.hideSplash(); this.palette?.toggle(); return }
    if (key.name === "escape") { this.chatInput.blur(); return }
    if (key.ctrl && key.name === "l") { this.softClear(); return }
    if (key.ctrl && key.name === "y") {
      const sel = this.renderer.getSelection()
      const text = sel?.getSelectedText()
      if (text) { this.renderer.copyToClipboardOSC52(text); this.showToast("copied to clipboard") }
      return
    }
    if (key.name === "i" && !key.ctrl && !key.meta) {
      if (!this.chatInput.focused) this.chatInput.focus()
    }
  }

  // ── Send ─────────────────────────────────────────────────────────

  async onSubmit(text: string): Promise<void> {
    if (!text.trim()) return
    if (text.startsWith("/")) { await this.handleSlash(text); return }
    if (this.streaming) return
    await this.sendQuery(text)
  }

  async sendQuery(text: string): Promise<void> {
    if (!text.trim() || this.streaming) return
    this.streaming = true
    this.hasStarted = true
    this.hideSplash()
    this.setMode("busy", th.fg.yellow)
    this.updateStatus("\u25CF agent thinking\u2026")
    this.addUserMsg(text)
    this.chatInput.value = ""
    this.chatInput.focus()
    this.proc?.stdin?.write(JSON.stringify({ query: text }) + "\n")
  }
}

// ─── Command Palette ─────────────────────────────────────────────────

class CommandPalette {
  renderer: CliRenderer
  app: A2ATuiApp
  box!: BoxRenderable
  searchInput!: InputRenderable
  listBox!: BoxRenderable
  rows: TextRenderable[] = []
  footerText!: TextRenderable
  visible = false
  commands: Array<{ id: string; label: string; desc: string; run: () => void }> = []
  filtered: typeof this.commands = []
  selectedIndex = 0

  constructor(r: CliRenderer, app: A2ATuiApp) {
    this.renderer = r; this.app = app; this.build()
  }

  build(): void {
    this.commands = [
      { id: "agents", label: "/agents", desc: "List connected agents",  run: () => { this.hide(); this.app.handleSlash("/agents") } },
      { id: "clear",  label: "/clear",  desc: "Clear conversation",     run: () => { this.hide(); this.app.handleSlash("/clear") } },
      { id: "help",   label: "/help",   desc: "Show commands",          run: () => { this.hide(); this.app.handleSlash("/help") } },
      { id: "exit",   label: "/exit",   desc: "Quit",                   run: () => { this.hide(); this.app.handleSlash("/exit") } },
    ]
    this.filtered = [...this.commands]
    const ph = Math.min(this.commands.length + 8, Math.max(10, this.renderer.height - 4))
    const pw = Math.min(60, Math.max(30, this.renderer.width - 4))

    this.box = new BoxRenderable(this.renderer, {
      id: "cmd-palette", width: pw, height: ph,
      position: "absolute", alignSelf: "center", top: 3, zIndex: 999,
      backgroundColor: th.bg.elevated, border: true, borderStyle: "rounded",
      borderColor: th.fg.accent, title: "\u2318 commands",
      titleColor: th.fg.accent, titleAlignment: "left", padding: 1, flexDirection: "column",
    })

    this.searchInput = new InputRenderable(this.renderer, {
      id: "palette-input", width: "100%",
      backgroundColor: th.bg.panel, focusedBackgroundColor: th.bg.panel,
      placeholder: "filter commands\u2026", placeholderColor: th.fg.muted,
      textColor: th.fg.primary, cursorColor: th.fg.accent,
    })
    this.searchInput.on(InputRenderableEvents.INPUT, (v: string) => this.onFilter(v))
    this.searchInput.on(InputRenderableEvents.ENTER, () => this.execute())
    this.box.add(this.searchInput)

    this.listBox = new BoxRenderable(this.renderer, {
      id: "palette-list", width: "100%", flexDirection: "column", marginTop: 1,
    })
    this.box.add(this.listBox)
    this.rebuildRows()

    this.footerText = new TextRenderable(this.renderer, {
      id: "palette-footer", content: "\u23CE run   \u2191\u2193 navigate   esc close",
      fg: th.fg.muted, marginTop: 1,
    })
    this.box.add(this.footerText)
    this.box.visible = false
    this.renderer.root.add(this.box)
  }

  private rebuildRows(): void {
    for (const r of this.rows) { this.listBox.remove(r); r.destroy() }
    this.rows = []
    if (this.filtered.length === 0) {
      const e = new TextRenderable(this.renderer, { content: "  nothing found", fg: th.fg.muted, width: "100%" })
      this.listBox.add(e); this.rows.push(e); return
    }
    this.filtered.forEach((c, i) => {
      const sel = i === this.selectedIndex
      const row = new TextRenderable(this.renderer, {
        content: ` ${sel ? "\u276F" : " "} ${c.label.padEnd(11)} ${c.desc}`,
        fg: sel ? th.fg.primary : th.fg.secondary,
        bg: sel ? th.bg.hover : th.bg.elevated, width: "100%",
      })
      this.listBox.add(row); this.rows.push(row)
    })
  }

  private onFilter(v: string): void {
    const q = v.toLowerCase()
    this.filtered = this.commands.filter(c =>
      c.label.toLowerCase().includes(q) || c.desc.toLowerCase().includes(q)
    )
    this.selectedIndex = 0; this.rebuildRows()
  }

  execute(): void { if (this.filtered.length) this.filtered[this.selectedIndex]!.run() }
  navigate(d: number): void {
    this.selectedIndex = Math.max(0, Math.min(this.filtered.length - 1, this.selectedIndex + d))
    this.rebuildRows()
  }
  show(): void {
    this.visible = true; this.filtered = [...this.commands]; this.selectedIndex = 0
    this.searchInput.value = ""; this.rebuildRows(); this.box.visible = true; this.searchInput.focus()
  }
  hide(): void { this.visible = false; this.box.visible = false; this.searchInput.blur(); this.app.chatInput.focus() }
  toggle(): void { this.visible ? this.hide() : this.show() }
}

// ─── Main ────────────────────────────────────────────────────────────

if (import.meta.main) {
  const renderer = await createCliRenderer({ exitOnCtrlC: true })
  const app = new A2ATuiApp()
  await app.build(renderer)

  renderer.keyInput.on("keypress", (key) => {
    if (key.name === "Escape" && !app.palette?.visible) {
      app.killBackend(); renderer.destroy(); process.exit(0)
    }
  })

  renderer.start()
  const projectDir = new URL("..", import.meta.url).pathname
  app.boot(projectDir)
}
