import { type SerializableValue, serialize } from "./stable-serialize";

// This function deep compares two objects for equality, regardless of their keys' order
export const equal = <T extends SerializableValue>(a: T, b: T): boolean => {
  return serialize(a) === serialize(b);
};
