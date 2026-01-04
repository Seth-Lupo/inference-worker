#!/usr/bin/env python3
"""
Decompose Microsoft custom ONNX ops to standard ops for TensorRT compatibility.

The ResembleAI chatterbox ONNX models use:
  - com.microsoft::MultiHeadAttention
  - com.microsoft::BiasGelu

This script replaces them with standard ONNX ops:
  - MultiHeadAttention → MatMul + Softmax + MatMul (scaled dot-product attention)
  - BiasGelu → Add + Gelu

Usage:
    python decompose_onnx.py --input model.onnx --output model_decomposed.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def decompose_with_graphsurgeon(input_path: str, output_path: str) -> bool:
    """
    Use ONNX GraphSurgeon to replace Microsoft custom ops with standard ops.
    """
    try:
        import onnx
        import onnx_graphsurgeon as gs

        print(f"Loading ONNX model: {input_path}")
        model = onnx.load(input_path)
        graph = gs.import_onnx(model)

        # Count ops before
        ms_ops_before = sum(1 for n in graph.nodes if n.domain == "com.microsoft")
        print(f"Microsoft ops before: {ms_ops_before}")

        nodes_to_remove = []
        nodes_to_add = []

        for node in graph.nodes:
            if node.domain != "com.microsoft":
                continue

            if node.op == "BiasGelu":
                # BiasGelu(input, bias) = Gelu(Add(input, bias))
                new_nodes = replace_bias_gelu(node, graph)
                if new_nodes:
                    nodes_to_remove.append(node)
                    nodes_to_add.extend(new_nodes)

            elif node.op == "MultiHeadAttention":
                # This is complex - for now, try to use standard Attention pattern
                new_nodes = replace_multihead_attention(node, graph)
                if new_nodes:
                    nodes_to_remove.append(node)
                    nodes_to_add.extend(new_nodes)

        # Remove old nodes and add new ones
        for node in nodes_to_remove:
            graph.nodes.remove(node)
        graph.nodes.extend(nodes_to_add)

        # Clean up
        graph.cleanup().toposort()

        # Count ops after
        ms_ops_after = sum(1 for n in graph.nodes if n.domain == "com.microsoft")
        print(f"Microsoft ops after: {ms_ops_after}")
        print(f"Replaced: {ms_ops_before - ms_ops_after} ops")

        # Export
        model = gs.export_onnx(graph)

        # Run shape inference to fix any issues
        try:
            from onnx import shape_inference
            model = shape_inference.infer_shapes(model)
        except Exception as e:
            print(f"Shape inference failed (non-fatal): {e}")

        onnx.save(model, output_path)
        print(f"Saved: {output_path}")

        return ms_ops_after == 0

    except Exception as e:
        print(f"GraphSurgeon decomposition failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def replace_bias_gelu(node, graph):
    """
    Replace BiasGelu with Add + GELU (decomposed to basic ops).

    BiasGelu(input, bias) -> Add(input, bias) then GELU

    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

    Decomposed to basic ops TensorRT supports:
      t1 = x * x * x
      t2 = x + 0.044715 * t1
      t3 = sqrt(2/π) * t2  (sqrt(2/π) ≈ 0.7978845608)
      t4 = tanh(t3)
      t5 = 1 + t4
      t6 = 0.5 * x * t5
    """
    import onnx_graphsurgeon as gs

    if len(node.inputs) < 2:
        print(f"  Skipping {node.name}: expected 2 inputs, got {len(node.inputs)}")
        return None

    input_tensor = node.inputs[0]
    bias_tensor = node.inputs[1]
    output_tensor = node.outputs[0]

    new_nodes = []

    # Constants
    const_044715 = gs.Constant(f"{node.name}_044715", values=np.array([0.044715], dtype=np.float32))
    const_sqrt2pi = gs.Constant(f"{node.name}_sqrt2pi", values=np.array([0.7978845608], dtype=np.float32))
    const_half = gs.Constant(f"{node.name}_half", values=np.array([0.5], dtype=np.float32))
    const_one = gs.Constant(f"{node.name}_one", values=np.array([1.0], dtype=np.float32))

    # x = input + bias
    x = gs.Variable(name=f"{node.name}_x")
    new_nodes.append(gs.Node(op="Add", name=f"{node.name}_add", inputs=[input_tensor, bias_tensor], outputs=[x]))

    # t1 = x * x
    x_sq = gs.Variable(name=f"{node.name}_x_sq")
    new_nodes.append(gs.Node(op="Mul", name=f"{node.name}_sq", inputs=[x, x], outputs=[x_sq]))

    # t2 = x * x * x = x_sq * x
    x_cube = gs.Variable(name=f"{node.name}_x_cube")
    new_nodes.append(gs.Node(op="Mul", name=f"{node.name}_cube", inputs=[x_sq, x], outputs=[x_cube]))

    # t3 = 0.044715 * x³
    t3 = gs.Variable(name=f"{node.name}_t3")
    new_nodes.append(gs.Node(op="Mul", name=f"{node.name}_mul_044715", inputs=[const_044715, x_cube], outputs=[t3]))

    # t4 = x + t3
    t4 = gs.Variable(name=f"{node.name}_t4")
    new_nodes.append(gs.Node(op="Add", name=f"{node.name}_add_t3", inputs=[x, t3], outputs=[t4]))

    # t5 = sqrt(2/π) * t4
    t5 = gs.Variable(name=f"{node.name}_t5")
    new_nodes.append(gs.Node(op="Mul", name=f"{node.name}_mul_sqrt2pi", inputs=[const_sqrt2pi, t4], outputs=[t5]))

    # t6 = tanh(t5)
    t6 = gs.Variable(name=f"{node.name}_t6")
    new_nodes.append(gs.Node(op="Tanh", name=f"{node.name}_tanh", inputs=[t5], outputs=[t6]))

    # t7 = 1 + t6
    t7 = gs.Variable(name=f"{node.name}_t7")
    new_nodes.append(gs.Node(op="Add", name=f"{node.name}_add_one", inputs=[const_one, t6], outputs=[t7]))

    # t8 = 0.5 * x
    t8 = gs.Variable(name=f"{node.name}_t8")
    new_nodes.append(gs.Node(op="Mul", name=f"{node.name}_mul_half", inputs=[const_half, x], outputs=[t8]))

    # output = t8 * t7 = 0.5 * x * (1 + tanh(...))
    new_nodes.append(gs.Node(op="Mul", name=f"{node.name}_final", inputs=[t8, t7], outputs=[output_tensor]))

    print(f"  Replaced BiasGelu: {node.name} ({len(new_nodes)} nodes)")
    return new_nodes


def replace_multihead_attention(node, graph):
    """
    Replace Microsoft MultiHeadAttention with standard ONNX ops.

    MS MHA inputs:
      0: query [batch, seq_q, hidden]
      1: key [batch, seq_k, hidden] (optional, defaults to query)
      2: value [batch, seq_v, hidden] (optional, defaults to key)
      3: bias (optional, packed QKV bias)
      4: key_padding_mask (optional)
      5: attention_bias (optional)

    MS MHA attributes:
      - num_heads: number of attention heads
      - scale: attention scale (usually 1/sqrt(head_dim))

    We decompose to standard scaled dot-product attention:
      Q, K, V = split(input) or use provided
      scores = softmax(Q @ K.T * scale)
      output = scores @ V
    """
    import onnx_graphsurgeon as gs

    # Get attributes
    num_heads = node.attrs.get("num_heads", 8)
    scale = node.attrs.get("scale", None)

    # Get inputs
    query = node.inputs[0]
    key = node.inputs[1] if len(node.inputs) > 1 and node.inputs[1].name else query
    value = node.inputs[2] if len(node.inputs) > 2 and node.inputs[2].name else key

    # Get output
    output = node.outputs[0]

    # For the chatterbox model, Q/K/V are already projected
    # We just need to reshape, do attention, and reshape back

    # Infer dimensions
    # Query shape is typically [batch, seq, num_heads * head_dim]
    q_shape = query.shape
    if q_shape is None or len(q_shape) < 3:
        print(f"  Skipping {node.name}: cannot infer query shape")
        return None

    # Try to get hidden dim
    hidden_dim = q_shape[-1] if isinstance(q_shape[-1], int) else None
    if hidden_dim is None:
        # Use scale to infer head_dim: scale = 1/sqrt(head_dim)
        if scale:
            head_dim = int(round(1.0 / (scale * scale)))
            hidden_dim = head_dim * num_heads
        else:
            hidden_dim = 512  # Default
            head_dim = hidden_dim // num_heads

    head_dim = hidden_dim // num_heads
    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    print(f"  MHA {node.name}: heads={num_heads}, hidden={hidden_dim}, head_dim={head_dim}, scale={scale}")

    # Create scale constant
    scale_const = gs.Constant(
        name=f"{node.name}_scale",
        values=np.array([scale], dtype=np.float32),
    )

    # Shape constants for reshape
    # [batch, seq, hidden] -> [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim]
    reshape_qkv_shape = gs.Constant(
        name=f"{node.name}_reshape_shape",
        values=np.array([0, -1, num_heads, head_dim], dtype=np.int64),
    )

    # For output: [batch, num_heads, seq, head_dim] -> [batch, seq, num_heads, head_dim] -> [batch, seq, hidden]
    reshape_out_shape = gs.Constant(
        name=f"{node.name}_reshape_out_shape",
        values=np.array([0, -1, hidden_dim], dtype=np.int64),
    )

    new_nodes = []

    # Reshape Q: [B, S, H] -> [B, S, heads, head_dim]
    q_reshaped = gs.Variable(name=f"{node.name}_q_reshaped")
    new_nodes.append(gs.Node(
        op="Reshape",
        name=f"{node.name}_reshape_q",
        inputs=[query, reshape_qkv_shape],
        outputs=[q_reshaped],
    ))

    # Transpose Q: [B, S, heads, head_dim] -> [B, heads, S, head_dim]
    q_transposed = gs.Variable(name=f"{node.name}_q_transposed")
    new_nodes.append(gs.Node(
        op="Transpose",
        name=f"{node.name}_transpose_q",
        inputs=[q_reshaped],
        outputs=[q_transposed],
        attrs={"perm": [0, 2, 1, 3]},
    ))

    # Reshape K
    k_reshaped = gs.Variable(name=f"{node.name}_k_reshaped")
    new_nodes.append(gs.Node(
        op="Reshape",
        name=f"{node.name}_reshape_k",
        inputs=[key, reshape_qkv_shape],
        outputs=[k_reshaped],
    ))

    # Transpose K: [B, S, heads, head_dim] -> [B, heads, head_dim, S] (for matmul)
    k_transposed = gs.Variable(name=f"{node.name}_k_transposed")
    new_nodes.append(gs.Node(
        op="Transpose",
        name=f"{node.name}_transpose_k",
        inputs=[k_reshaped],
        outputs=[k_transposed],
        attrs={"perm": [0, 2, 3, 1]},  # Note: different from Q to get K^T
    ))

    # Reshape V
    v_reshaped = gs.Variable(name=f"{node.name}_v_reshaped")
    new_nodes.append(gs.Node(
        op="Reshape",
        name=f"{node.name}_reshape_v",
        inputs=[value, reshape_qkv_shape],
        outputs=[v_reshaped],
    ))

    # Transpose V: [B, S, heads, head_dim] -> [B, heads, S, head_dim]
    v_transposed = gs.Variable(name=f"{node.name}_v_transposed")
    new_nodes.append(gs.Node(
        op="Transpose",
        name=f"{node.name}_transpose_v",
        inputs=[v_reshaped],
        outputs=[v_transposed],
        attrs={"perm": [0, 2, 1, 3]},
    ))

    # Q @ K^T: [B, heads, S_q, head_dim] @ [B, heads, head_dim, S_k] -> [B, heads, S_q, S_k]
    qk_matmul = gs.Variable(name=f"{node.name}_qk")
    new_nodes.append(gs.Node(
        op="MatMul",
        name=f"{node.name}_matmul_qk",
        inputs=[q_transposed, k_transposed],
        outputs=[qk_matmul],
    ))

    # Scale: scores = QK * scale
    scores_scaled = gs.Variable(name=f"{node.name}_scores_scaled")
    new_nodes.append(gs.Node(
        op="Mul",
        name=f"{node.name}_scale",
        inputs=[qk_matmul, scale_const],
        outputs=[scores_scaled],
    ))

    # Softmax over last axis
    attn_weights = gs.Variable(name=f"{node.name}_attn_weights")
    new_nodes.append(gs.Node(
        op="Softmax",
        name=f"{node.name}_softmax",
        inputs=[scores_scaled],
        outputs=[attn_weights],
        attrs={"axis": -1},
    ))

    # Attention @ V: [B, heads, S_q, S_k] @ [B, heads, S_k, head_dim] -> [B, heads, S_q, head_dim]
    attn_output = gs.Variable(name=f"{node.name}_attn_out")
    new_nodes.append(gs.Node(
        op="MatMul",
        name=f"{node.name}_matmul_v",
        inputs=[attn_weights, v_transposed],
        outputs=[attn_output],
    ))

    # Transpose back: [B, heads, S, head_dim] -> [B, S, heads, head_dim]
    output_transposed = gs.Variable(name=f"{node.name}_out_transposed")
    new_nodes.append(gs.Node(
        op="Transpose",
        name=f"{node.name}_transpose_out",
        inputs=[attn_output],
        outputs=[output_transposed],
        attrs={"perm": [0, 2, 1, 3]},
    ))

    # Reshape: [B, S, heads, head_dim] -> [B, S, hidden]
    new_nodes.append(gs.Node(
        op="Reshape",
        name=f"{node.name}_reshape_out",
        inputs=[output_transposed, reshape_out_shape],
        outputs=[output],
    ))

    print(f"  Replaced MultiHeadAttention: {node.name} ({len(new_nodes)} nodes)")
    return new_nodes


def check_custom_ops(model_path: str) -> list:
    """Check what custom ops are in the model."""
    import onnx

    model = onnx.load(model_path)
    custom_ops = {}

    for node in model.graph.node:
        if node.domain and node.domain not in ('', 'ai.onnx', 'ai.onnx.ml'):
            key = f"{node.domain}::{node.op_type}"
            custom_ops[key] = custom_ops.get(key, 0) + 1

    return custom_ops


def decompose_with_raw_onnx(input_path: str, output_path: str) -> bool:
    """
    Fallback: Use raw ONNX manipulation without graphsurgeon.
    This is simpler but less robust.
    """
    try:
        import onnx
        from onnx import helper, TensorProto

        print(f"Loading ONNX model: {input_path}")
        model = onnx.load(input_path)
        graph = model.graph

        # Find Microsoft ops
        ms_nodes = [(i, node) for i, node in enumerate(graph.node)
                    if node.domain == "com.microsoft"]

        if not ms_nodes:
            print("No Microsoft ops found")
            onnx.save(model, output_path)
            return True

        print(f"Found {len(ms_nodes)} Microsoft ops")

        # For each MS op, clear the domain to try standard ONNX
        # This won't work for all ops but might for some
        modified = False
        for idx, node in ms_nodes:
            if node.op_type == "BiasGelu":
                # BiasGelu can sometimes work as standard Gelu with preprocessing
                # For now, just try clearing domain
                node.domain = ""
                node.op_type = "Gelu"  # Hope ONNX has it
                print(f"  Converted BiasGelu -> Gelu: {node.name}")
                modified = True
            elif node.op_type == "MultiHeadAttention":
                # Can't easily convert MHA without graphsurgeon
                print(f"  Cannot convert MultiHeadAttention without graphsurgeon: {node.name}")

        if modified:
            onnx.save(model, output_path)
            print(f"Saved (partial conversion): {output_path}")

        return False  # Partial conversion, likely still has issues

    except Exception as e:
        print(f"Raw ONNX manipulation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Decompose Microsoft ONNX custom ops for TensorRT"
    )
    parser.add_argument("--input", "-i", required=True, help="Input ONNX file")
    parser.add_argument("--output", "-o", required=True, help="Output ONNX file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Check what custom ops we're dealing with
    print(f"Analyzing: {input_path}")
    custom_ops = check_custom_ops(str(input_path))

    if not custom_ops:
        print("No custom ops found - model should be TRT compatible")
        import shutil
        shutil.copy(input_path, output_path)
        sys.exit(0)

    print(f"Custom ops found:")
    for op, count in custom_ops.items():
        print(f"  {op}: {count}")

    # Try graphsurgeon first
    print(f"\nDecomposing custom ops with GraphSurgeon...")
    success = decompose_with_graphsurgeon(str(input_path), str(output_path))

    if not success:
        print(f"\nGraphSurgeon failed, trying raw ONNX manipulation...")
        success = decompose_with_raw_onnx(str(input_path), str(output_path))

    if success:
        # Verify
        remaining = check_custom_ops(str(output_path))
        if remaining:
            print(f"\nWarning: Some custom ops remain:")
            for op, count in remaining.items():
                print(f"  {op}: {count}")
            sys.exit(1)
        else:
            print(f"\nSuccess! All custom ops decomposed.")
            print(f"Output: {output_path}")
    else:
        print(f"\nDecomposition failed.")
        print(f"\nTo fix, try:")
        print(f"  pip3.12 uninstall onnx onnx-graphsurgeon")
        print(f"  pip3.12 install onnx==1.15.0 onnx-graphsurgeon==0.5.2")
        sys.exit(1)


if __name__ == "__main__":
    main()
