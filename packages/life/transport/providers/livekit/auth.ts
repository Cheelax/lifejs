import { AccessToken } from "livekit-server-sdk";
import type { GetTokenFunction } from "@/transport/auth";

export const getToken: GetTokenFunction<"livekit"> = async (config, roomName, participantId) => {
  // Create a token with the room name and participant name
  const token = new AccessToken(config.apiKey, config.apiSecret, {
    identity: participantId,
  });

  token.addGrant({
    roomJoin: true,
    room: roomName,
    canPublish: true,
    canSubscribe: true,
    canPublishData: true,
    roomCreate: true,
  });

  return await token.toJwt();
};
