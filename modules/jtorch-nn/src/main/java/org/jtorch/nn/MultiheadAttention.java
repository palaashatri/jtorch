package org.jtorch.nn;

import org.jtorch.api.Torch;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;

/**
 * Multi-head scaled dot-product attention without masks or dropout. Parameters use PyTorch's
 * {@code in_proj_weight}/{@code in_proj_bias}/{@code out_proj} layout so state dictionaries are interchangeable
 * for the supported configuration.
 */
public final class MultiheadAttention extends Module {
    private final int embedDim;
    private final int heads;
    private final int headDim;
    private final Parameter inProjWeight;
    private final Parameter inProjBias;
    private final Linear outProj;

    public MultiheadAttention(int embedDim, int heads) {
        this(embedDim, heads, true);
    }

    public MultiheadAttention(int embedDim, int heads, boolean bias) {
        if (embedDim <= 0 || heads <= 0 || embedDim % heads != 0) {
            throw new IllegalArgumentException("MultiheadAttention requires embedDim divisible by heads");
        }
        this.embedDim = embedDim;
        this.heads = heads;
        this.headDim = embedDim / heads;
        this.inProjWeight = registerParameter("in_proj_weight",
                new Parameter(Init.uniform(Shape.of(3 * embedDim, embedDim), embedDim)));
        this.inProjBias = bias
                ? registerParameter("in_proj_bias", new Parameter(Torch.zeros(Shape.of(3 * embedDim))))
                : null;
        this.outProj = registerModule("out_proj", new Linear(embedDim, embedDim, bias));
    }

    /** Self-attention over a batch-first {@code [batch, sequence, embedDim]} tensor. */
    public Tensor forward(Tensor input) {
        return forward(input, input, input);
    }

    public Tensor forward(Tensor query, Tensor key, Tensor value) {
        if (query.dim() != 3 || key.dim() != 3 || value.dim() != 3) {
            throw new org.jtorch.core.ShapeException(
                    "MultiheadAttention expects batch-first [batch, sequence, embedDim] tensors");
        }
        if (query.size(2) != embedDim || key.size(2) != embedDim || value.size(2) != embedDim) {
            throw new org.jtorch.core.ShapeException("MultiheadAttention input embedding width must be " + embedDim);
        }
        if (query.size(0) != key.size(0) || query.size(0) != value.size(0)
                || key.size(1) != value.size(1)) {
            throw new org.jtorch.core.ShapeException("MultiheadAttention query/key/value batch shapes do not agree");
        }
        Tensor qWeight = inProjWeight.slice(0, 0, embedDim, 1);
        Tensor kWeight = inProjWeight.slice(0, embedDim, 2 * embedDim, 1);
        Tensor vWeight = inProjWeight.slice(0, 2 * embedDim, 3 * embedDim, 1);
        Tensor qBias = inProjBias == null ? null : inProjBias.slice(0, 0, embedDim, 1);
        Tensor kBias = inProjBias == null ? null : inProjBias.slice(0, embedDim, 2 * embedDim, 1);
        Tensor vBias = inProjBias == null ? null : inProjBias.slice(0, 2 * embedDim, 3 * embedDim, 1);
        Tensor q = query.linear(qWeight, qBias);
        Tensor k = key.linear(kWeight, kBias);
        Tensor v = value.linear(vWeight, vBias);
        long batch = q.size(0);
        long sequence = q.size(1);
        q = q.reshape(batch, sequence, heads, headDim).permute(0, 2, 1, 3);
        k = k.reshape(batch, key.size(1), heads, headDim).permute(0, 2, 1, 3);
        v = v.reshape(batch, value.size(1), heads, headDim).permute(0, 2, 1, 3);
        Tensor scores = q.matmul(k.transpose(2, 3)).mul(1.0 / Math.sqrt(headDim));
        Tensor attention = scores.softmax(3);
        Tensor context = attention.matmul(v).permute(0, 2, 1, 3).reshape(batch, sequence, embedDim);
        return outProj.call(context);
    }

    public int embedDim() {
        return embedDim;
    }

    public int heads() {
        return heads;
    }
}
