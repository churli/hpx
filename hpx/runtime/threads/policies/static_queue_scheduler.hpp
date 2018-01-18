//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_SCHEDULING_STATIC_QUEUE_JUL_22_2015_0103PM)
#define HPX_THREADMANAGER_SCHEDULING_STATIC_QUEUE_JUL_22_2015_0103PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_STATIC_SCHEDULER)
#include <hpx/compat/mutex.hpp>
#include <hpx/runtime/threads/policies/local_queue_scheduler.hpp>
#include <hpx/runtime/threads/policies/lockfree_queue_backends.hpp>
#include <hpx/runtime/threads/policies/thread_queue.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/threads_fwd.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/logging.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

// TODO: add branch prediction and function heat

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies
{
#ifdef HPX_HAVE_THREAD_MINIMAL_DEADLOCK_DETECTION
    ///////////////////////////////////////////////////////////////////////////
    // We globally control whether to do minimal deadlock detection using this
    // global bool variable. It will be set once by the runtime configuration
    // startup code
    extern bool minimal_deadlock_detection;
#endif

    ///////////////////////////////////////////////////////////////////////////
    /// The local_queue_scheduler maintains exactly one queue of work
    /// items (threads) per OS thread, where this OS thread pulls its next work
    /// from.
    template <typename Mutex = compat::mutex,
        typename PendingQueuing = lockfree_fifo,
        typename StagedQueuing = lockfree_fifo,
        typename TerminatedQueuing = lockfree_lifo>
    class static_queue_scheduler
        : public local_queue_scheduler<
            Mutex, PendingQueuing, StagedQueuing, TerminatedQueuing
          >
    {
    public:
        typedef local_queue_scheduler<
            Mutex, PendingQueuing, StagedQueuing, TerminatedQueuing
        > base_type;

        static_queue_scheduler(
                typename base_type::init_parameter_type const& init,
                bool deferred_initialization = true)
          : base_type(init, deferred_initialization)
        {}

        static std::string get_scheduler_name()
        {
            return "static_queue_scheduler";
        }

        void suspend(std::size_t)
        {
            HPX_ASSERT_MSG(false, "static_queue_scheduler does not support"
                " suspending");
        }

        void resume(std::size_t)
        {
            HPX_ASSERT_MSG(false, "static_queue_scheduler does not support"
                " resuming");
        }

        /// Return the next thread to be executed, return false if none is
        /// available
        virtual bool get_next_thread(std::size_t num_thread, bool,
            std::int64_t& idle_loop_count, threads::thread_data*& thrd)
        {
            typedef typename base_type::thread_queue_type thread_queue_type;

            std::size_t queues_size = this->queues_.size();

            {
                HPX_ASSERT(num_thread < queues_size);

                thread_queue_type* q = this->queues_[num_thread];
                bool result = q->get_next_thread(thrd);

                q->increment_num_pending_accesses();
                if (result)
                    return true;
                q->increment_num_pending_misses();
            }

            return false;
        }
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif
#endif

