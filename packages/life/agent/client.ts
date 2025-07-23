import { z } from "zod";
import type { ClientConfig } from "@/config/client";
import { TransportClient } from "@/transport/client";

export const agentClientConfigSchema = z.object({});

export class AgentClient {
  transport: TransportClient;
  config: ClientConfig;

  constructor(config: ClientConfig) {
    this.config = config;
    this.transport = new TransportClient(config.transport);
  }

  async inviteAgent(roomId: string, token: string) {
    await this.transport.joinRoom(roomId, token);
    await this.transport.enableMicrophone();
    await this.transport.playAudio();
  }

  async say(text: string) {
    await this.transport.sendObject("rpc-say", { text });
  }
}
