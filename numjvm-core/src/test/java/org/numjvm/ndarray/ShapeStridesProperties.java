package org.numjvm.ndarray;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;

import net.jqwik.api.Arbitrary;
import net.jqwik.api.ForAll;
import net.jqwik.api.Property;
import net.jqwik.api.Provide;

import org.numjvm.testing.ArrayArbitraries;
import org.numjvm.testing.ArrayArbitraries.Layout;

class ShapeStridesProperties {

    @Provide
    Arbitrary<Shape> shapes() {
        return ArrayArbitraries.shapes(6, 1000);
    }

    @Provide
    Arbitrary<Shape> smallShapes() {
        return ArrayArbitraries.smallShapes();
    }

    @Provide
    Arbitrary<Integer> itemSizes() {
        return ArrayArbitraries.itemSizes();
    }

    @Provide
    Arbitrary<Layout> layouts() {
        return ArrayArbitraries.layouts();
    }

    @Property
    void sizeIsTheProductOfDimensions(@ForAll("shapes") Shape shape) {
        long product = 1;
        for (long d : shape.dimensions()) {
            product *= d;
        }
        assertEquals(product, shape.size());
    }

    @Property
    void contiguousStridesAreContiguousInTheirOrder(@ForAll("shapes") Shape shape, @ForAll("itemSizes") int item) {
        assertTrue(Strides.isCContiguous(shape, Strides.cContiguous(shape, item), item));
        assertTrue(Strides.isFContiguous(shape, Strides.fContiguous(shape, item), item));
    }

    @Property
    void contiguousExtentCoversExactlyTheElements(@ForAll("shapes") Shape shape, @ForAll("itemSizes") int item) {
        ByteExtent c = Strides.extent(shape, Strides.cContiguous(shape, item), 0, item);
        ByteExtent f = Strides.extent(shape, Strides.fContiguous(shape, item), 0, item);
        assertEquals(shape.size() * item, c.length());
        assertEquals(shape.size() * item, f.length());
    }

    @Property
    void extentEqualsMinAndMaxOfEnumeratedOffsets(@ForAll("layouts") Layout layout) {
        ByteExtent e = Strides.extent(layout.shape(), layout.strides(), layout.offset(), layout.itemSize());
        long[] offsets = ArrayArbitraries.enumerateOffsets(layout);
        if (offsets.length == 0) {
            assertTrue(e.isEmpty());
            return;
        }
        long min = Arrays.stream(offsets).min().orElseThrow();
        long max = Arrays.stream(offsets).max().orElseThrow();
        assertEquals(min, e.start(), layout::toString);
        assertEquals(max + layout.itemSize(), e.end(), layout::toString);
        assertTrue(e.start() >= 0, "generator keeps every layout inside its buffer");
    }

    @Property
    void cContiguousMeansOffsetsAreConsecutive(@ForAll("layouts") Layout layout) {
        int item = layout.itemSize();
        boolean c = Strides.isCContiguous(layout.shape(), layout.strides(), item);
        long[] offsets = ArrayArbitraries.enumerateOffsets(layout);
        boolean consecutive = true;
        for (int k = 1; k < offsets.length; k++) {
            consecutive &= offsets[k] == offsets[k - 1] + item;
        }
        if (c) {
            assertTrue(consecutive, layout::toString);
        }
        if (consecutive && offsets.length > 0) {
            assertTrue(c, layout::toString);
        }
    }

    @Property
    void equalShapesHaveEqualHashCodes(@ForAll("smallShapes") Shape shape) {
        Shape copy = Shape.of(shape.dimensions());
        assertEquals(shape, copy);
        assertEquals(shape.hashCode(), copy.hashCode());
    }
}
