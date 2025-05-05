package bucket;

import java.util.ArrayList;
import java.util.List;


public class Main {
    public static void main(String[] args) {
        var src = new ArrayList<Integer>(List.of(3, 6, 2, 8, 5, 1, 4, 7));

        int maxValue = src.stream().max(Integer::compareTo).orElse(0);
        var bucketSort = new BucketSort();
        var results = bucketSort.bucketSort(src, maxValue + 1);

        System.out.println("Before: " + src);
        System.out.println("After : " + results);
        System.out.println("DONE");
    }
}