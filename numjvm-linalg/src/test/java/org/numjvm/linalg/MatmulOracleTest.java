package org.numjvm.linalg;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.math.BigDecimal;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import org.numjvm.dtype.DType;
import org.numjvm.iteration.IndexIterator;
import org.numjvm.ndarray.Shape;
import static org.junit.jupiter.api.Assertions.*;

class MatmulOracleTest {
    private static long[] shape(String s) { return s.equals("-")?new long[0]:Arrays.stream(s.split(",")).mapToLong(Long::parseLong).toArray(); }
    private static double[] values(String s) { return s.equals("-")?new double[0]:Arrays.stream(s.split(",")).mapToDouble(Double::parseDouble).toArray(); }
    @Test void numpyShapesDtypesAndValues() throws Exception {
        int count=0;
        try(var reader=new BufferedReader(new InputStreamReader(getClass().getResourceAsStream("/matmul-oracle.tsv"),StandardCharsets.UTF_8))) {
            for(String line:reader.lines().toList()) {
                if(line.startsWith("#"))continue;
                String[] f=line.split("\t");
                var a=NJ.array(values(f[5]),shape(f[2])).astype(f[0].equals("float32")?DType.FLOAT32:DType.FLOAT64);
                var b=NJ.array(values(f[6]),shape(f[3])).astype(f[1].equals("float32")?DType.FLOAT32:DType.FLOAT64);
                var result=Linalg.matmul(a,b);assertEquals(Shape.of(shape(f[4])),result.shape());
                assertEquals(a.dtype()==DType.FLOAT32 && b.dtype()==DType.FLOAT32?DType.FLOAT32:DType.FLOAT64,result.dtype());
                var expected=values(f[7]);var it=new IndexIterator(result.shape());int i=0;
                while(it.hasNext())assertEquals(expected[i++],result.getDouble(it.next()),result.dtype()==DType.FLOAT32?2e-6:2e-14);
                count++;
            }
        }
        assertEquals(24,count);
    }
    @Test void cancellationIsCheckedAgainstExactBinaryInputReference() {
        for(DType type:new DType[]{DType.FLOAT32,DType.FLOAT64}) {
            var a=NJ.array(new double[]{1e10,1,-1e10,3,1e-10}).astype(type);
            var b=NJ.array(new double[]{1,2,1,-2,1}).astype(type);
            BigDecimal exact=BigDecimal.ZERO,scale=BigDecimal.ZERO;
            for(int i=0;i<5;i++) {
                BigDecimal term=new BigDecimal(a.getDouble(i)).multiply(new BigDecimal(b.getDouble(i)));
                exact=exact.add(term);scale=scale.add(term.abs());
            }
            double u=type==DType.FLOAT32?Math.scalb(1.0,-24):Math.scalb(1.0,-53);
            double gamma=5*u/(1-5*u);
            double actual=Linalg.matmul(a,b).getDouble();
            assertTrue(new BigDecimal(actual).subtract(exact).abs().doubleValue()<=gamma*scale.doubleValue()+Double.MIN_VALUE);
        }
    }
    @Test void exceptionalValuesAndTransposeIdentityHaveSeparateExpectations() {
        assertTrue(Double.isNaN(Linalg.matmul(NJ.array(new double[]{0}),NJ.array(new double[]{Double.POSITIVE_INFINITY})).getDouble()));
        assertEquals(Double.POSITIVE_INFINITY,Linalg.matmul(NJ.array(new double[]{1}),NJ.array(new double[]{Double.POSITIVE_INFINITY})).getDouble());
        var a=NJ.arange(0,12,1).astype(DType.FLOAT64).reshape(3,4);var b=NJ.ones(4,2);
        var c=Linalg.matmul(a,b).transpose();var d=Linalg.matmul(b.transpose(),a.transpose());
        for(int i=0;i<2;i++)for(int j=0;j<3;j++)assertEquals(c.getDouble(i,j),d.getDouble(i,j));
    }
    @Test void subnormalAndOverflowCasesHaveExplicitExpectations() {
        for(DType type:new DType[]{DType.FLOAT32,DType.FLOAT64}) {
            double tiny=type==DType.FLOAT32?Float.MIN_VALUE:Double.MIN_VALUE;
            double large=type==DType.FLOAT32?Float.MAX_VALUE:Double.MAX_VALUE;
            assertEquals(tiny,Linalg.matmul(NJ.array(new double[]{tiny}).astype(type),NJ.ones(type,1)).getDouble());
            assertEquals(Double.POSITIVE_INFINITY,Linalg.matmul(NJ.array(new double[]{large}).astype(type),NJ.array(new double[]{2}).astype(type)).getDouble());
        }
        var broadcast=NJ.ones(1,3).broadcastTo(2,3);
        var result=Linalg.matmul(broadcast,NJ.ones(3,2));
        assertEquals(3,result.getDouble(1,1));
    }
}
