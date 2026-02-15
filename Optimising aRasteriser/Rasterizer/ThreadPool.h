#pragma once

#include <vector>
#include <queue>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

// Thread pool: fixed number of workers; submitBatch(jobs), waitIdle().
class ThreadPool {
public:
    explicit ThreadPool(size_t numWorkers)
        : numWorkers_(numWorkers)
        , pending_(0)
        , stop_(false)
    {
        workers_.reserve(numWorkers_);
        for (size_t i = 0; i < numWorkers_; i++) {
            workers_.emplace_back(&ThreadPool::workerLoop, this);
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_)
            if (t.joinable()) t.join();
    }

    // Submit a batch of jobs; do not modify jobs or captured refs until waitIdle().
    void submitBatch(const std::vector<std::function<void()>>& jobs) {
        if (jobs.empty()) return;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto& j : jobs)
                queue_.push(j);
            pending_ += static_cast<int>(jobs.size());
        }
        cv_.notify_all();
    }

    // Block until all jobs in the current batch have finished.
    void waitIdle() {
        std::unique_lock<std::mutex> lock(mutex_);
        idle_.wait(lock, [this] { return pending_ == 0; });
    }

private:
    void workerLoop() {
        while (true) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
                if (stop_ && queue_.empty()) break;
                if (!queue_.empty()) {
                    job = std::move(queue_.front());
                    queue_.pop();
                }
            }
            if (job) {
                job();
                std::lock_guard<std::mutex> lock(mutex_);
                pending_--;
                if (pending_ == 0)
                    idle_.notify_all();
            }
        }
    }

    size_t numWorkers_;
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable idle_;
    std::atomic<int> pending_;
    bool stop_;
};
