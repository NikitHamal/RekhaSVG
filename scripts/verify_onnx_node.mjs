#!/usr/bin/env node
/**
 * Node.js ONNX Runtime verification script (scripts/verify_onnx_node.mjs)
 * Verifies rekha_net_int8.onnx and rekha_text_int8.onnx execution in Node.js:
 *   - nan_count === 0
 *   - CPU inference < 600 ms
 */
import fs from "node:fs";
import path from "node:path";

async function main() {
  const modelDir = process.argv[2] || "/work/rekhasvg/models";
  const netPath = path.join(modelDir, "rekha_net_int8.onnx");
  const txtPath = path.join(modelDir, "rekha_text_int8.onnx");

  const netMb = fs.statSync(netPath).size / (1024 * 1024);
  const txtMb = fs.statSync(txtPath).size / (1024 * 1024);

  const ort = await import("onnxruntime-node");
  const opts = { intraOpNumThreads: 4, interOpNumThreads: 1 };
  const sessNet = await ort.InferenceSession.create(netPath, opts);
  const sessTxt = await ort.InferenceSession.create(txtPath, opts);

  const imgData = new Float32Array(1 * 3 * 512 * 512);
  for (let i = 0; i < imgData.length; i++) imgData[i] = 0.5;
  const imgTensor = new ort.Tensor("float32", imgData, [1, 3, 512, 512]);

  const cropData = new Float32Array(1 * 1 * 32 * 192);
  for (let i = 0; i < cropData.length; i++) cropData[i] = 0.5;
  const cropTensor = new ort.Tensor("float32", cropData, [1, 1, 32, 192]);

  // Warmup
  await sessNet.run({ image: imgTensor });
  await sessTxt.run({ crop: cropTensor });

  const t0 = performance.now();
  const resNet = await sessNet.run({ image: imgTensor });
  const netMs = performance.now() - t0;

  const t1 = performance.now();
  const resTxt = await sessTxt.run({ crop: cropTensor });
  const txtMs = performance.now() - t1;

  const outNet = resNet.logits_7ch.data;
  const outTxt = resTxt.log_probs.data;
  let nanCount = 0;
  for (let i = 0; i < outNet.length; i++) if (Number.isNaN(outNet[i])) nanCount++;
  for (let i = 0; i < outTxt.length; i++) if (Number.isNaN(outTxt[i])) nanCount++;

  const report = {
    runtime: "onnxruntime-node",
    rekha_net_int8_mb: Number(netMb.toFixed(2)),
    rekha_text_int8_mb: Number(txtMb.toFixed(2)),
    total_int8_mb: Number((netMb + txtMb).toFixed(2)),
    rekha_net_node_ms: Number(netMs.toFixed(2)),
    rekha_text_node_ms: Number(txtMs.toFixed(2)),
    total_node_ms: Number((netMs + txtMs).toFixed(2)),
    nan_count: nanCount,
  };
  console.log(JSON.stringify(report, null, 2));
  if (nanCount !== 0 || netMs + txtMs >= 600) {
    process.exit(1);
  }
}

main();
