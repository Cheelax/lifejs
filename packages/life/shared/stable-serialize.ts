import superjson from "superjson";
import { ZodError, z } from "zod";

// Register custom transformer for ZodError to preserve all error information
// We use unknown[] as the serialized type since ZodIssue has complex union types
// that don't satisfy SuperJSON's JSONValue constraints
// biome-ignore lint/suspicious/noExplicitAny: z.ZodIssue[] is not a valid JSONValue, but is serializable
superjson.registerCustom<ZodError, any>(
  {
    isApplicable: (v): v is ZodError => v instanceof ZodError,
    serialize: (zErr) => zErr.issues,
    deserialize: (data) => new ZodError(data as z.ZodIssue[]),
  },
  "ZodError",
);

// - Primitive types
const serializablePrimitivesSchema = z.union([
  z.string(),
  z.number(),
  z.boolean(),
  z.null(),
  z.undefined(),
  z.bigint(),
  z.date(),
  z.instanceof(RegExp),
  z.instanceof(Error),
  z.instanceof(URL),
  z.instanceof(ArrayBuffer),
  z.instanceof(Int8Array),
  z.instanceof(Uint8Array),
  z.instanceof(Uint8ClampedArray),
  z.instanceof(Int16Array),
  z.instanceof(Uint16Array),
  z.instanceof(Int32Array),
  z.instanceof(Uint32Array),
  z.instanceof(Float32Array),
  z.instanceof(Float64Array),
  z.instanceof(BigInt64Array),
  z.instanceof(BigUint64Array),
]);
type SerializablePrimitives = z.infer<typeof serializablePrimitivesSchema>;

// Collections and recursive types
export const serializableValueSchema: z.ZodType<SerializableValue> = z.lazy(() =>
  z.union([
    serializablePrimitivesSchema,
    z.array(serializableValueSchema),
    z.set(serializableValueSchema),
    z.map(z.any(), serializableValueSchema),
    z.record(z.string(), serializableValueSchema),
  ]),
);
export type SerializableValue =
  | SerializablePrimitives
  | SerializableValue[]
  | Set<SerializableValue>
  | Map<SerializableValue, SerializableValue>
  | { [key: string]: SerializableValue };

// Serialize and deserialize functions using SuperJSON
export const serialize = (value: SerializableValue): string => {
  // - Serialize with SuperJSON
  const superJsonOutput = superjson.stringify(value);
  const serializedObject = JSON.parse(superJsonOutput);

  // - Stringify again with stable keys order
  return stableDeepStringify(serializedObject);
};

export const deserialize = <T extends SerializableValue = SerializableValue>(value: string): T => {
  return superjson.parse<T>(value);
};

/**
 * This function stringifies a given object into a JSON string, producing an
 * output with a stable keys order compared to JSON.stringify().
 *
 * Source: fast-json-stable-stringify (https://github.com/epoberezkin/fast-json-stable-stringify/blob/master/index.js)
 */

// biome-ignore-start lint/style: reason
// biome-ignore-start lint/suspicious: reason
// biome-ignore-start lint/complexity: reason
// biome-ignore-start lint/correctness: reason

export function stableDeepStringify(data: any, opts?: any): string {
  if (!opts) opts = {};
  if (typeof opts === "function") opts = { cmp: opts };
  var cycles = typeof opts.cycles === "boolean" ? opts.cycles : false;

  var cmp =
    opts.cmp &&
    ((f) => (node: any) => (a: any, b: any) => {
      var aobj = { key: a, value: node[a] };
      var bobj = { key: b, value: node[b] };
      return f(aobj, bobj);
    })(opts.cmp);

  var seen: any[] = [];
  return (function stringify(node: any) {
    if (node && node.toJSON && typeof node.toJSON === "function") {
      node = node.toJSON();
    }

    if (node === undefined) return "";
    if (typeof node == "number") return isFinite(node) ? "" + node : "null";
    if (typeof node !== "object") return JSON.stringify(node);

    var i, out;
    if (Array.isArray(node)) {
      out = "[";
      for (i = 0; i < node.length; i++) {
        if (i) out += ",";
        out += stringify(node[i]) || "null";
      }
      return out + "]";
    }

    if (node === null) return "null";

    if (seen.indexOf(node) !== -1) {
      if (cycles) return JSON.stringify("__cycle__");
      throw new TypeError("Converting circular structure to JSON");
    }

    var seenIndex = seen.push(node) - 1;
    var keys = Object.keys(node).sort(cmp && cmp(node));
    out = "";
    for (i = 0; i < keys.length; i++) {
      var key = keys[i];
      var value = stringify(node[key!]);

      if (!value) continue;
      if (out) out += ",";
      out += JSON.stringify(key) + ":" + value;
    }
    seen.splice(seenIndex, 1);
    return "{" + out + "}";
  })(data);
}

// biome-ignore-end lint/style: reason
// biome-ignore-end lint/suspicious: reason
// biome-ignore-end lint/complexity: reason
// biome-ignore-end lint/correctness: reason
