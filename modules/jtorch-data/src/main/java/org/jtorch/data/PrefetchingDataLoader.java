package org.jtorch.data;

import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Bounded prefetch wrapper around a batch source. The producer runs on a virtual thread because this is I/O and
 * orchestration, never numerical tiling. Callers that stop early should close the wrapper to interrupt the producer.
 */
public final class PrefetchingDataLoader implements Iterable<Sample>, AutoCloseable {
    private static final Object END = new Object();

    private final Iterable<Sample> source;
    private final int depth;
    private volatile boolean closed;
    private final List<Thread> workers = new CopyOnWriteArrayList<>();

    public PrefetchingDataLoader(Iterable<Sample> source, int depth) {
        if (depth <= 0) {
            throw new IllegalArgumentException("prefetch depth must be positive");
        }
        this.source = source;
        this.depth = depth;
    }

    @Override
    public Iterator<Sample> iterator() {
        BlockingQueue<Object> queue = new ArrayBlockingQueue<>(depth);
        Thread producer = Thread.ofVirtual().name("jtorch-prefetch").start(() -> {
            try {
                for (Sample sample : source) {
                    if (closed) {
                        return;
                    }
                    queue.put(sample);
                }
            } catch (InterruptedException error) {
                Thread.currentThread().interrupt();
            } catch (Throwable error) {
                queue.offer(new Failure(error));
            } finally {
                queue.offer(END);
            }
        });
        workers.add(producer);
        return new Iterator<>() {
            private Sample next;
            private boolean fetched;
            private boolean done;

            private void fetch() {
                if (fetched || done) {
                    return;
                }
                try {
                    Object value = queue.take();
                    if (value == END) {
                        done = true;
                    } else if (value instanceof Failure failure) {
                        done = true;
                        throw new IllegalStateException("prefetch producer failed", failure.error());
                    } else {
                        next = (Sample) value;
                        fetched = true;
                    }
                } catch (InterruptedException error) {
                    Thread.currentThread().interrupt();
                    done = true;
                    throw new IllegalStateException("prefetch interrupted", error);
                }
            }

            @Override
            public boolean hasNext() {
                fetch();
                return !done;
            }

            @Override
            public Sample next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                fetched = false;
                Sample value = next;
                next = null;
                return value;
            }
        };
    }

    @Override
    public void close() {
        closed = true;
        for (Thread producer : workers) {
            producer.interrupt();
        }
    }

    private record Failure(Throwable error) {}
}
