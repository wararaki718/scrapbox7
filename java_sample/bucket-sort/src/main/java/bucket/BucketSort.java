package bucket;
import java.util.ArrayList;
import java.util.Collections;


public class BucketSort {
    public ArrayList<Integer> bucketSort(ArrayList<Integer> src, int range) {
        ArrayList<Integer> buckets = new ArrayList<>(Collections.nCopies(range, 0));
        ArrayList<Integer> offsets = new ArrayList<>(Collections.nCopies(1, 0));
        ArrayList<Integer> results = new ArrayList<>(Collections.nCopies(src.size(), 0));

        for (var num : src) {
            buckets.set(num, buckets.get(num) + 1);
        }

        for (int i = 0; i < range-1; i++) {
            offsets.add(offsets.get(i) + buckets.get(i));
        }
        
        for (var target: src) {
            var value = offsets.get(target);
            results.set(value, target);
            offsets.set(target, value + 1);
        }

        return results;
    }
}