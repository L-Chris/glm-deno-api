import { createParser } from 'eventsource-parser'
import { OpenAI, GLM } from './types.ts'
import { approximateTokenSize } from 'tokenx'
import { uuid } from "./utils.ts";
import { parseAssistantMessage } from "./assistant-message/index.ts";

export class ChunkTransformer {
  private streamController!: ReadableStreamDefaultController
  private stream: ReadableStream
  private encoder = new TextEncoder()
  private decoder = new TextDecoder()
  private content = ''
  private config: OpenAI.ChatConfig
  private isThinking = false
  private messages: OpenAI.Message[] = []
  private citations: string[] = []
  private sentBlockIndex = -1
  private parser = createParser({
    onEvent: e => {
      this.parse(e)
    }
  })
  private callbacks: (() => void)[] = []

  constructor (req: Response, config: OpenAI.ChatConfig, messages: OpenAI.Message[]) {
    this.messages = messages
    this.config = config
    this.stream = new ReadableStream({
      start: controller => {
        this.streamController = controller
        this.read(req)
      }
    })
  }

  // 根据对接模型修改
  private parse (e: EventSourceMessage) {
    if (!e.data) return
    const chunkData: GLM.CompletionChunk = JSON.parse(e.data)
    const chunkType = this.getChunkType(chunkData)
    if (chunkType === CHUNK_TYPE.NONE) return
    const part = chunkData.parts[0]
    const content = part?.content?.[0]

    switch (chunkType) {
      case CHUNK_TYPE.TEXT: {
        if (!content.text) return
        if (this.isThinking) {
          this.isThinking = false
        }
        const deltaText = this.config.is_incremental_chunk ? content.text : content.text.slice(this.content.length)
        this.send({ 
          content: deltaText
        })
        break
      }
      case CHUNK_TYPE.THINKING: {
        this.isThinking = true
        const thinkContent = content.think || content.text || ''
        const deltaText = this.config.is_incremental_chunk ? thinkContent : thinkContent.slice(this.content.length)
        this.send({ 
          reasoning_content: deltaText
        })
        break
      }
      // 有可能触发多次，在结束前发送即可
      case CHUNK_TYPE.SEARCHING_DONE: {
        this.citations = (part.meta_data.metadata_list || []).map(_ => _.url)
        this.send({ citations: this.citations, content: this.isThinking ? '' : content.content, reasoning_content: this.isThinking ? content.content : '' })
        break
      }
      case CHUNK_TYPE.START:
        if (chunkData.conversation_id) {
          this.config.chat_id = chunkData.conversation_id
        }
        break
    }
  }

  // 根据对接模型修改
  private getChunkType (chunk: GLM.CompletionChunk) {
    const part = chunk.parts[0]
    const content = part?.content?.[0]
    if (chunk.parts.length === 0 && !this.content) return CHUNK_TYPE.START
    if (!part || !content || part.status === 'finish') return CHUNK_TYPE.NONE
    if (content.type === 'think' || content.type === 'text_thinking') return CHUNK_TYPE.THINKING
    if (part.meta_data.citations) return CHUNK_TYPE.DONE
    if (content.type === 'text') return CHUNK_TYPE.TEXT
    if (content.type === 'browser_result') return CHUNK_TYPE.SEARCHING_DONE
    return CHUNK_TYPE.NONE
  }

  private async read (req: Response) {
    if (!this.streamController) return
    try {
      const contentType = req.headers.get('content-type') || ''
      if (contentType.indexOf('text/event-stream') < 0) {
        const body = await req.text()
        this.send({ error: contentType === 'text/html' ? 'rejected by server' : body })
        this.send({ done: true })
        return
      }

      const reader = req.body!.getReader()
      while (true) {
        const { done, value } = await reader.read()
        // console.log('read', done, decodedValue)
        if (done) {
          this.send({ done: true })
          return
        }
        const decodedValue = this.decoder.decode(value)
        // console.log(decodedValue)
        this.parser.feed(decodedValue)
      }
    } catch (err) {
      this.send({ error: err instanceof Error ? err.message : 'unknown error' })
      this.send({ done: true })
    }
  }

  private send (params: {
    content?: string
    citations?: string[]
    reasoning_content?: string
    error?: string
    done?: boolean
  }) {
    this.content += (params.reasoning_content || '') + (params.content || '')
    const message: OpenAI.CompletionChunk = {
      id: '',
      model: this.config.model_name,
      object: 'chat.completion.chunk',
      choices: [{
        index: 0,
        delta: {
            role: 'assistant',
            content: params.content || '',
            reasoning_content: params.reasoning_content || ''
        },
        finish_reason: null
      }],
      citations: params.citations || [],
      created: Math.trunc(Date.now() / 1000)
    }

    if (params.error) {
        message.error = {
            message: params.error,
            type: 'server error'
        }

        this.streamController.enqueue(this.encoder.encode(`data: ${JSON.stringify(message)}\n\n`))
        return
    }

    if (this.config.tools?.length > 0) {
      const blocks = parseAssistantMessage(this.content)
      const block = blocks[this.sentBlockIndex + 1]
      // 只发送完整的块
      if (block && !block.partial) {
        if (block.type === 'text') {
          const thinkingOpenTagIndex = block.content.indexOf('<thinking>')
          const thinkingCloseTagIndex = block.content.indexOf('</thinking>')
          if (thinkingOpenTagIndex >= 0 && thinkingCloseTagIndex >= 0) {
            message.choices[0].delta!.content = block.content.slice(thinkingCloseTagIndex + 11)
            message.choices[0].delta!.reasoning_content = block.content.slice(thinkingOpenTagIndex + 10, thinkingCloseTagIndex)
          } else {
            message.choices[0].delta!.content = block.content
            message.choices[0].delta!.reasoning_content = ''
          }
        } else if (block.type === 'tool_use') {
          message.choices[0].delta!.content = ''
          message.choices[0].delta!.reasoning_content = ''
          message.choices[0].delta!.tool_calls = [
            {
              id: uuid(),
              type: 'function',
              function: {
                name: block.params.tool_name!,
                arguments: block.params.arguments || ''
              }
            }
          ]
          message.choices[0].finish_reason = 'tool_calls'
        }
        // Deno.writeFileSync(`./data/${this.config.chat_id}_res_1.json`, new TextEncoder().encode(JSON.stringify(message)))
        this.streamController.enqueue(this.encoder.encode(`data: ${JSON.stringify(message)}\n\n`))
        this.sentBlockIndex++
        message.choices[0].delta!.content = ''
        message.choices[0].delta!.reasoning_content = ''
        message.choices[0].delta!.tool_calls = []
      }
    }

    if (params.done) {
        if (this.config.tools?.length > 0 && this.sentBlockIndex === -1) {
          message.choices[0].delta!.content = this.content
        }
        const prompt_tokens = approximateTokenSize(this.messages.reduce((acc, cur) => acc + (Array.isArray(cur.content) ? cur.content.map(_ => _.text).join('') : cur.content), ''))
        const completion_tokens = approximateTokenSize(this.content)
        message.usage = {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens   
        }
        message.choices[0].finish_reason = 'stop'
        this.streamController.enqueue(this.encoder.encode(`data: ${JSON.stringify(message)}\n\n`))
        // Deno.writeFileSync(`./data/${this.config.chat_id}_res_2.json`, new TextEncoder().encode(JSON.stringify(message)))
        this.streamController.enqueue(this.encoder.encode(`data: [DONE]\n\n`))
        this.streamController.close()
        this.callbacks.forEach(cb => cb())
        return
    }

    if (this.config.tools?.length === 0) {
      this.streamController.enqueue(this.encoder.encode(`data: ${JSON.stringify(message)}\n\n`))
    }
  }

  onDone(cb: () => void) {
    this.callbacks.push(cb)
  }

  getStream() {
    return this.stream
  }

  formatLink(desc: string) {
    return desc.replace(/\[(\d+(?:,\d+)*)\]\(@ref\)/g, (_: string, numbers: string) => {
      return numbers.split(',').map((n: string) => `[${n}]`).join('')
    })
  }
}

interface EventSourceMessage {
  data: string
  event?: string
  id?: string
}

export enum CHUNK_TYPE {
  ERROR = 'ERROR',
  START = 'START', // 提供基础信息，如chatid
  DEEPSEARCHING = 'DEEPSEARCHING',
  SEARCHING = 'SEARCHING',
  SEARCHING_DONE = 'SEARCHING_DONE',
  THINKING = 'THINKING',
  TEXT = 'TEXT',
  SUGGESTION = 'SUGGESTION',
  DONE = 'DONE',
  NONE = 'NONE'
}
