export declare namespace OpenAI {
  interface Message {
    role: 'user' | 'assistant' | 'system' | 'tool'
    type?: 'text' | 'image' | 'file'
    image?: string
    content: string | MessageContent[]
    tool_calls?: ToolCall[]
    tool_call_id?: string
  }

  interface MessageContent {
    type: 'text' | 'file' | 'image_url' | 'image'
    file_url?: {
      url: string
    }
    image_url?: {
      url: string
    }
    text?: string
  }

  interface CompletionChunk {
    id: string
    model: string
    object: string
    citations: string[]
    created: number
    choices: {
      index: number
      message?: {
        role: 'assistant' | 'user'
        content: string
        reasoning_content?: string
        tool_calls?: ToolCall[]
      }
      delta?: {
        role: 'assistant' | 'user'
        content: string
        reasoning_content?: string
        tool_calls?: ToolCall[]
      }
      finish_reason: null | 'stop' | 'tool_calls'
    }[]
    usage?: {
      prompt_tokens: number
      completion_tokens: number
      total_tokens: number
    }
    error?: {
      message: string
      type: string
    }
  }

  interface Tool {
    type: 'function'
    function: {
      name: string
      description: string
      parameters: {
        type: 'object'
        properties: Record<
          string,
          {
            type: string
            description: string
          }
        >
        required: string[]
        additionalProperties: boolean
      }
      strict: boolean
    }
  }

  interface ToolCall {
    id: string
    type: 'function'
    function: {
      name: string
      arguments: string
    }
  }

  type ToolChoice = 'auto' | 'required'

  interface ChatConfig {
    chat_id: string
    chat_type: 't2t' | 't2v' | 't2i' | 'search' | 'artifacts'
    model_name: string
    response_format: {
      type: 'text' | 'json_schema'
      json_schema?: Record<string, any>
    }
    features: {
      thinking: boolean
      searching: boolean
      deepsearching: boolean
    }
    stream: boolean
    is_incremental_chunk: boolean
    tools: Tool[]
    tool_choice: ToolChoice
    is_tool_calling: boolean
    is_tool_calling_done: boolean
  }
}

export declare namespace GLM {
  interface Message {
    role: OpenAI.Message['role']
    content: string | MessageContent[]
  }

  interface MessageContent {
    type: 'text' | 'file' | 'image_url' | 'image'
    file_url?: {
      url: string
    }
    image_url?: {
      url: string
    }
    text?: string
  }

  interface Attachment {
    id: string
    user_id: string
    hash: string | null
    filename: string
    data: Record<string, any>
    meta: {
      name: string
      content_type: string
      size: number
    }
    created_at: number
    updated_at: number
    type: 'image' | 'file'
  }

  interface AttachmentResponse {
    type: 'image' | 'file'
    id: string
    url: string
    image: string
    name: string
    status: 'uploaded'
    size: number
    error: string
    itemId: string
    file_type: string
    showType: 'image' | 'file'
  }

  type CompletionChunk = {
    id: string
    conversation_id: string
    assistant_id: string
    parts: {
      id: string
      logic_id: string
      role: 'assistant' | 'user' | 'tool'
      content: {
        type: 'text' | 'think' | 'text_thinking' | 'tool_calls' | 'browser_result'
        text: string
        tool_calls?: {
          name: 'search' | 'open' | 'finish'
          arguments: string
        }
        content?: string
        think?: string
        third_distribute: Record<string, any>
        agent_distribute: Record<string, any>
      }[]
      model: string
      recipient: string
      meta_data: {
        citations?: {
          citation_format_type: string
          end_ix: number
          metadata: {
            extra: {
              cited_message_idx: number
              evidence_text: string
            }
            text: string
            title: string
            type: string
            url: string
          }
          start_ix: number
        }[]
        metadata_list?: SearchResult[]
        auto_glm_data: Record<string, any>
        think_finished_time?: number
      }
      status: 'finish' | 'init'
    }[]
    created_at: string
    status: 'init' | 'finish'
    last_error: Record<string, any>
    meta_data: {
      input_question_type: string
      if_plus_model: boolean
      plus_model_available: boolean
      if_increase_push: boolean
    }
  }

  interface SearchResult {
    text: string
    title: string
    type: 'webpage'
    url: string
    icon_link: string
    media: string
  }

  interface Model {
    id: string;
    name: string;
    object: string;
    created: number;
    owned_by: string;
    preset: boolean;
    action_ids: string[];
    info: {
      id: string;
      user_id: string;
      base_model_id: null;
      name: string;
      meta: {
        profile_image_url: string;
        description: string;
        short_description: string;
        max_context_length: number;
        max_generation_length: number;
        is_single_round?: number;
        chat_type: string[];
        modality: string[];
        capabilities: {
          vision: boolean;
          document: boolean;
          video: boolean;
          citations: boolean;
        };
      };
      params: {
        model_type: string;
        max_ref_token: number;
        max_input_tokens: number;
        enable_reasoning_content?: boolean;
        system?: string;
        seed?: number;
        ignore_single_turn?: boolean;
      };
      access_control: null;
      is_active: boolean;
      is_visitor_active: boolean;
      updated_at: number;
      created_at: number;
    };
  }
}
