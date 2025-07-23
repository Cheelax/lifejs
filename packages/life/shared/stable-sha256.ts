import type { SerializableValue } from "./stable-serialize";
import { serialize } from "./stable-serialize";

// This function deep compares two objects for equality, regardless of their keys' order
export const sha256 = async (obj: SerializableValue) => {
  const json = serialize(obj);
  const hashedData = new TextEncoder().encode(json);
  const hashBuffer = await crypto.subtle.digest("SHA-256", hashedData);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
};
