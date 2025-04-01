import { uuid, extractJsonFromContent, unixTimestamp } from './utils.ts'
import { OpenAI, GLM } from './types.ts'
import { approximateTokenSize } from 'tokenx'
import { ChunkTransformer } from './chunk-transformer.ts'
import { parseAssistantMessage } from './assistant-message/parse-assistant-message.ts'

export async function removeConversation (convId: string, ticket: string) {
  const refreshToken = await acquireToken(ticket)
  if (!refreshToken) return
  return fetch(`https://chatglm.cn/chatglm/backend-api/assistant/conversation/delete`, {
    method: 'POST',
    headers: generateHeaders(refreshToken),
    body: JSON.stringify({
      assistant_id: '65940acff94777010aa6b796',
      conversation_id: convId
    })
  })
}

export async function createCompletionStream (
  params: {
    messages: GLM.Message[]
    config: OpenAI.ChatConfig
    token: string
  },
  callback = () => {}
) {
  const refreshToken = await acquireToken(params.token)
  if (!refreshToken) return
  const req = await fetch(
    `https://chatglm.cn/chatglm/backend-api/assistant/stream`,
    {
      method: 'POST',
      headers: {
        ...generateHeaders(refreshToken),
        Accept: 'text/event-stream'
      },
      body: JSON.stringify({
        assistantId: '65940acff94777010aa6b796',
        conversation_id: params.config.chat_id,
        messages: params.messages,
        meta_data: {
          is_test: false,
          input_question_type: 'xxxx',
          channel: '',
          draft_id: '',
          is_networking: false,
          chat_mode: params.config.features.deepsearching ? 'deep_research' : params.config.features.thinking ? 'zero' : '',
          quote_log_id: '',
          if_increase_push: true,
          platform: 'pc'
        }
      })
    }
  )

  const parser = new ChunkTransformer(req, params.config, params.messages)

  parser.onDone(callback)

  return parser.getStream()
}

export async function createCompletion (params: {
  messages: GLM.Message[]
  config: OpenAI.ChatConfig
  token: string
}) {
  const config = params.config
  const isJson = params.config.response_format.type === 'json_schema'
  const lastMessage = params.messages.findLast(_ => _.role === 'user')!
  const refreshToken = await acquireToken(params.token)
  if (!refreshToken) return

  if (isJson) {
    // 如果是JSON格式，添加特殊指令
    const schema = params.config.response_format?.json_schema
      ? `\n按照以下JSON Schema格式返回：\n${JSON.stringify(
          params.config.response_format.json_schema,
          null,
          2
        )}`
      : '\n请以有效的JSON格式返回响应。'

    lastMessage.content = `${
      Array.isArray(lastMessage.content)
        ? lastMessage.content[0].text
        : lastMessage.content
    }${schema}`
  }

  const req = await fetch(`https://chatglm.cn/chatglm/backend-api/assistant/stream`, {
    method: 'POST',
    headers: generateHeaders(refreshToken),
    body: JSON.stringify({
      assistantId: '65940acff94777010aa6b796',
      conversation_id: params.config.chat_id,
      messages: params.messages,
      meta_data: {
        is_test: false,
        input_question_type: 'xxxx',
        channel: '',
        draft_id: '',
        is_networking: false,
        chat_mode: params.config.features.deepsearching ? 'deep_research' : params.config.features.thinking ? 'zero' : '',
        quote_log_id: '',
        if_increase_push: true,
        platform: 'pc'
      }
    })
  })

  const body = await req.json()

  const message: OpenAI.CompletionChunk = {
    id: '',
    model: config.model_name,
    object: 'chat.completion',
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: body?.choices?.[0]?.message?.content || '',
          tool_calls: []
        },
        finish_reason: 'stop'
      }
    ],
    citations: [] as string[],
    usage: {
      prompt_tokens: 1,
      completion_tokens: 1,
      total_tokens: 2
    },
    created: Math.trunc(Date.now() / 1000)
  }

  return formatMessageResponse(
    message,
    params.messages,
    config.response_format,
    config.tools
  )
}

export async function getModels (params: { token: string }) {
  return [
    {
      id: 'glm4',
      name: "glm4",
    },
    {
      id: 'glm4_think',
      name: "glm4_think",
    },
    {
      id: 'glm4_deepsearch',
      name: "glm4_deepsearch",
    },
  ]
}

function formatMessageResponse (
  message: OpenAI.CompletionChunk,
  promptMessages: OpenAI.Message[],
  response_format?: OpenAI.ChatConfig['response_format'],
  tools?: OpenAI.Tool[]
) {
  if (!message.choices[0].message) return message

  const prompt = promptMessages.reduce(
    (acc, cur) =>
      acc +
      (Array.isArray(cur.content)
        ? cur.content.map(_ => _.text).join('')
        : cur.content),
    ''
  )
  const prompt_tokens = approximateTokenSize(prompt)
  const completion_tokens = approximateTokenSize(
    message.choices[0].message.content
  )

  message.usage = {
    prompt_tokens: prompt_tokens,
    completion_tokens: completion_tokens,
    total_tokens: prompt_tokens + completion_tokens
  }
  if (response_format?.type === 'json_schema') {
    const json = extractJsonFromContent(message.choices[0].message.content)
    if (json) {
      message.choices[0].message.content = JSON.stringify(json)
    }
  }

  if (!tools?.length) return message
  const blocks = parseAssistantMessage(message.choices[0].message.content)
  message.choices[0].message.content = blocks
    .filter(_ => _.type === 'text')
    .map(_ => _.content)
    .join('')
  message.choices[0].message.tool_calls = blocks
    .filter(_ => _.type === 'tool_use')
    .map(_ => ({
      id: uuid(),
      type: 'function',
      function: {
        name: _.params.tool_name!,
        arguments: _.params.arguments || ''
      }
    }))

  return message
}

// access_token映射
const accessTokenMap = new Map();
// access_token请求队列映射
const accessTokenRequestQueueMap: Record<string, Function[]> = {};

/**
 * 请求access_token
 *
 * 使用refresh_token去刷新获得access_token
 *
 * @param refreshToken 用于刷新access_token的refresh_token
 */
async function requestToken(refreshToken: string) {
  if (accessTokenRequestQueueMap[refreshToken])
    return new Promise((resolve) =>
      accessTokenRequestQueueMap[refreshToken].push(resolve)
    );
  accessTokenRequestQueueMap[refreshToken] = [];
  
  try {
    const result = await fetch(
      "https://chatglm.cn/chatglm/backend-api/v1/user/refresh",
      {
        method: 'post',
        headers: generateHeaders(refreshToken),
        body: JSON.stringify({})
      }
    );

    const contentType = result.headers.get('content-type') || ''
    if (contentType === 'text/html') {
      console.error('rejected by server')
      throw new Error('rejected by server')
    }
    const body: {
      code: number
      status: string
      message: string
      result: any
    } = await result.json()

    console.log(body)

    if (body.code !== 0 || !body.result?.accessToken) {
      throw new Error(body.message)
    }

    const token = body?.result?.accessToken

    if (accessTokenRequestQueueMap[refreshToken]) {
      accessTokenRequestQueueMap[refreshToken].forEach((resolve) =>
        resolve(result)
      );
      delete accessTokenRequestQueueMap[refreshToken];
    }

    return {
      accessToken: token,
      refreshToken: refreshToken,
      refreshTime: unixTimestamp() + 3600,
    }
  } catch(err) {
    if (accessTokenRequestQueueMap[refreshToken]) {
      accessTokenRequestQueueMap[refreshToken].forEach((resolve) =>
        resolve(err)
      );
      delete accessTokenRequestQueueMap[refreshToken];
    }
  }
}

/**
 * 获取缓存中的access_token
 *
 * 避免短时间大量刷新token，未加锁，如果有并发要求还需加锁
 *
 * @param refreshToken 用于刷新access_token的refresh_token
 */
async function acquireToken(refreshToken: string): Promise<string> {
  let result = accessTokenMap.get(refreshToken);
  if (!result) {
    result = await requestToken(refreshToken);
    accessTokenMap.set(refreshToken, result);
  }
  if (unixTimestamp() > result.refreshTime) {
    result = await requestToken(refreshToken);
    accessTokenMap.set(refreshToken, result);
  }
  return result.accessToken?.token;
}

export function generateHeaders (token: string) {
  return {
    Authorization: `Bearer ${token}`,
    Referer: "https://chatglm.cn/main/alltoolsdetail",
    "X-Device-Id": uuid(),
    "X-Request-Id": uuid(),
    Accept: "*/*",
    "App-Name": "chatglm",
    'X-App-Platform': "pc",
    'X-App-Version': '0.0.1',
    'X-Device-Brand': '',
    'X-Device-Model': '',
    Origin: "https://chatglm.cn",
    "Sec-Ch-Ua":
      '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "User-Agent":
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
  }
}
