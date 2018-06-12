//  Copyright (c) 2007-2018 Hartmut Kaiser
//  Copyright (c) 2014-2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_CONTINUATION_APR_17_2012_0150PM)
#define HPX_LCOS_LOCAL_CONTINUATION_APR_17_2012_0150PM

#include <hpx/config.hpp>
#include <hpx/error_code.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/traits/future_traits.hpp>
#if defined(HPX_HAVE_EXECUTOR_COMPATIBILITY)
#include <hpx/traits/v1/is_executor.hpp>
#endif
#include <hpx/util/annotated_function.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/thread_description.hpp>

#include <hpx/parallel/executors/execution.hpp>
#include <hpx/parallel/executors/post_policy_dispatch.hpp>

#include <boost/intrusive_ptr.hpp>

#include <exception>
#include <functional>
#include <mutex>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail
{
    template <typename Future>
    struct transfer_result
    {
        template <typename Source, typename Destination>
        static void apply(Source && src, Destination& dest, std::false_type)
        {
            try {
                dest.set_value(src.get());
            }
            catch (...) {
                dest.set_exception(std::current_exception());
            }
        }

        template <typename Source, typename Destination>
        static void apply(Source && src, Destination& dest, std::true_type)
        {
            try {
                src.get();
                dest.set_value(util::unused);
            }
            catch (...) {
                dest.set_exception(std::current_exception());
            }
        }

        template <typename Source, typename DestinationState>
        void operator()(Source && src, DestinationState const& dest) const
        {
            typedef std::is_void<
                typename traits::future_traits<Future>::type
            > is_void;

            apply(std::forward<Source>(src), *dest, is_void());
        }
    };

    template <typename Func, typename Future, typename Continuation>
    void invoke_continuation(Func& func, Future && future, Continuation& cont,
        std::false_type)
    {
        try {
            cont.set_value(func(std::forward<Future>(future)));
        }
        catch (...) {
            cont.set_exception(std::current_exception());
        }
    }

    template <typename Func, typename Future, typename Continuation>
    void invoke_continuation(Func& func, Future && future, Continuation& cont,
        std::true_type)
    {
        try {
            func(std::forward<Future>(future));
            cont.set_value(util::unused);
        }
        catch (...) {
            cont.set_exception(std::current_exception());
        }
    }

    template <typename Func, typename Future, typename Continuation>
    typename std::enable_if<
       !traits::detail::is_unique_future<
            typename util::invoke_result<Func, Future>::type
        >::value
    >::type
    invoke_continuation(Func& func, Future && future, Continuation& cont)
    {
        typedef std::is_void<
            typename util::invoke_result<Func, Future>::type
        > is_void;

        hpx::util::annotate_function annotate(func);
        invoke_continuation(func, std::forward<Future>(future), cont, is_void());
    }

    template <typename Func, typename Future, typename Continuation>
    typename std::enable_if<
        traits::detail::is_unique_future<
            typename util::invoke_result<Func, Future>::type
        >::value
    >::type
    invoke_continuation(Func& func, Future && future, Continuation& cont)
    {
        try {
            typedef
                typename util::invoke_result<Func, Future>::type
                inner_future;
            typedef
                typename traits::detail::shared_state_ptr_for<inner_future>::type
                inner_shared_state_ptr;

            // take by value, as the future may go away immediately
            auto inner = func(std::forward<Future>(future));
            inner_shared_state_ptr inner_state =
                traits::detail::get_shared_state(inner);

            typename inner_shared_state_ptr::element_type* ptr =
                inner_state.get();
            if (ptr == nullptr && !inner.is_ready())
            {
                HPX_THROW_EXCEPTION(no_state,
                    "invoke_continuation",
                    "the inner future has no valid shared state");
            }

            // Bind an on_completed handler to this future which will transfer
            // its result to the new future.
            boost::intrusive_ptr<Continuation> cont_(&cont);
            if (ptr != nullptr)
            {
                ptr->execute_deferred();
                ptr->set_on_completed(
                    util::deferred_call(transfer_result<inner_future>{},
                        std::move(inner), std::move(cont_)));
            }
            else
            {
                transfer_result<inner_future>{}(
                    std::move(inner), std::move(cont_));
            }
        }
        catch (...) {
            cont.set_exception(std::current_exception());
        }
     }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ContResult>
    struct continuation_result
    {
        typedef ContResult type;
    };

    template <typename ContResult>
    struct continuation_result<future<ContResult> >
    {
        typedef ContResult type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename F, typename ContResult>
    class continuation
      : public future_data<ContResult>
    {
    private:
        typedef future_data<ContResult> base_type;

        typedef typename base_type::mutex_type mutex_type;
        typedef typename base_type::result_type result_type;

    protected:
        threads::thread_id_type get_id() const
        {
            std::lock_guard<mutex_type> l(this->mtx_);
            return id_;
        }
        void set_id(threads::thread_id_type const& id)
        {
            std::lock_guard<mutex_type> l(this->mtx_);
            id_ = id;
        }

        struct reset_id
        {
            reset_id(continuation& target)
              : target_(target)
            {
                if (threads::get_self_ptr() != nullptr)
                    target.set_id(threads::get_self_id());
            }
            ~reset_id()
            {
                target_.set_id(threads::invalid_thread_id);
            }
            continuation& target_;
        };

    public:
        typedef typename base_type::init_no_addref init_no_addref;

        template <typename Func, typename Enable = typename
            std::enable_if<
                !std::is_same<typename hpx::util::decay<Func>::type,
                    continuation>::value>::type>
        continuation(Func && f)
          : started_(false), id_(threads::invalid_thread_id)
          , f_(std::forward<Func>(f))
        {}

        template <typename Func>
        continuation(init_no_addref no_addref, Func && f)
          : base_type(no_addref),
            started_(false), id_(threads::invalid_thread_id),
            f_(std::forward<Func>(f))
        {}

    protected:
        void run_impl(Future && future)
        {
            invoke_continuation(f_, std::move(future), *this);
        }

    public:
        void run(Future && f, threads::thread_priority, error_code& ec)
        {
            {
                std::lock_guard<mutex_type> l(this->mtx_);
                if (started_) {
                    HPX_THROWS_IF(ec, task_already_started,
                        "continuation::run",
                        "this task has already been started");
                    return;
                }
                started_ = true;
            }

            run_impl(std::move(f));

            if (&ec != &throws)
                ec = make_success_code();
        }

        void run(Future && f, threads::thread_priority priority)
        {
            run(std::move(f), priority, throws);
        }

    protected:
        threads::thread_result_type
        async_impl(Future && future)
        {
            reset_id r(*this);

            invoke_continuation(f_, std::move(future), *this);
            return threads::thread_result_type(threads::terminated,
                    threads::invalid_thread_id);
        }

        threads::thread_result_type
        async_exec_impl(Future && future)
        {
            typedef std::is_void<
                    typename util::invoke_result<F, Future>::type
                > is_void;

            reset_id r(*this);

            invoke_continuation(f_, std::move(future), *this, is_void());
            return threads::thread_result_type(threads::terminated,
                threads::invalid_thread_id);
        }

    public:
        void async(Future && f, threads::thread_priority priority,
            error_code& ec)
        {
            {
                std::unique_lock<mutex_type> l(this->mtx_);
                if (started_) {
                    l.unlock();
                    HPX_THROWS_IF(ec, task_already_started,
                        "continuation::async",
                        "this task has already been started");
                    return;
                }
                started_ = true;
            }

            boost::intrusive_ptr<continuation> this_(this);
            threads::thread_result_type (continuation::*async_impl_ptr)(
                Future &&) = &continuation::async_impl;

            hpx::util::thread_description desc(async_impl_ptr,
                "hpx::parallel::execution::parallel_executor::post");

            parallel::execution::detail::post_policy_dispatch<
                    hpx::launch::async_policy
                >::call(desc, hpx::launch::async, async_impl_ptr,
                    std::move(this_), std::move(f));

            if (&ec != &throws)
                ec = make_success_code();
        }

        void async(Future && f, threads::thread_priority priority)
        {
            async(std::move(f), priority, throws);
        }

#if defined(HPX_HAVE_EXECUTOR_COMPATIBILITY)
        template <typename Executor>
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        void async_exec_v1(Future && f, Executor& exec, error_code& ec)
        {
            {
                std::unique_lock<mutex_type> l(this->mtx_);
                if (started_) {
                    l.unlock();
                    HPX_THROWS_IF(ec, task_already_started,
                        "continuation::async_exec_v1",
                        "this task has already been started");
                    return;
                }
                started_ = true;
            }

            boost::intrusive_ptr<continuation> this_(this);
            threads::thread_result_type (continuation::*async_impl_ptr)(
                Future &&) = &continuation::async_impl;

            parallel::executor_traits<Executor>::apply_execute(exec,
                util::one_shot(async_impl_ptr), std::move(this_), std::move(f));

            if (&ec != &throws)
                ec = make_success_code();
        }

        template <typename Executor>
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        void async_exec_v1(Future && f, Executor& exec)
        {
            async_exec_v1(std::move(f), exec, throws);
        }
#endif

        template <typename Executor>
        void async_exec(Future && f, Executor && exec, error_code& ec)
        {
            {
                std::unique_lock<mutex_type> l(this->mtx_);
                if (started_) {
                    l.unlock();
                    HPX_THROWS_IF(ec, task_already_started,
                        "continuation::async_exec",
                        "this task has already been started");
                    return;
                }
                started_ = true;
            }

            boost::intrusive_ptr<continuation> this_(this);
            threads::thread_result_type (continuation::*async_exec_impl_ptr)(
                Future &&) = &continuation::async_exec_impl;

            parallel::execution::post(std::forward<Executor>(exec),
                async_exec_impl_ptr, std::move(this_), std::move(f));

            if (&ec != &throws)
                ec = make_success_code();
        }

        template <typename Executor>
        void async_exec(Future && f, Executor && exec)
        {
            async_exec(std::move(f), std::forward<Executor>(exec), throws);
        }

        // cancellation support
        bool cancelable() const
        {
            return true;
        }

        void cancel()
        {
            std::unique_lock<mutex_type> l(this->mtx_);
            try {
                if (!this->started_)
                    HPX_THROW_THREAD_INTERRUPTED_EXCEPTION();

                if (this->is_ready_locked(l))
                    return;   // nothing we can do

                if (id_ != threads::invalid_thread_id) {
                    // interrupt the executing thread
                    threads::interrupt_thread(id_);

                    this->started_ = true;

                    l.unlock();
                    this->set_error(future_cancelled,
                        "continuation<Future, ContResult>::cancel",
                        "future has been canceled");
                }
                else {
                    l.unlock();
                    HPX_THROW_EXCEPTION(future_can_not_be_cancelled,
                        "continuation<Future, ContResult>::cancel",
                        "future can't be canceled at this time");
                }
            }
            catch (...) {
                this->started_ = true;
                this->set_exception(std::current_exception());
                throw;
            }
        }

    public:
        template <typename Policy>
        void attach(Future && fut, Policy && policy)
        {
            typedef typename std::decay<Future>::type future_type;
            typedef typename traits::detail::shared_state_ptr_for<
                future_type>::type shared_state_ptr;

            // bind an on_completed handler to this future which will invoke
            // the continuation
            boost::intrusive_ptr<continuation> this_(this);

            shared_state_ptr state = traits::detail::get_shared_state(fut);
            typename shared_state_ptr::element_type* ptr = state.get();

            if (ptr == nullptr && !fut.is_ready())
            {
                HPX_THROW_EXCEPTION(no_state,
                    "continuation::attach",
                    "the future to attach has no valid shared state");
            }

            auto cont =
                [HPX_CAPTURE_MOVE(this_)](future_type && f, launch policy)
                {
                    if (hpx::detail::has_async_policy(policy))
                        this_->async(std::move(f), policy.priority());
                    else
                        this_->run(std::move(f), policy.priority());
                };

            if (ptr != nullptr)
            {
                ptr->execute_deferred();

                auto future = hpx::traits::future_access<future_type>::create(
                    std::move(state));
                ptr->set_on_completed(
                    util::deferred_call(std::move(cont), std::move(future),
                        std::forward<Policy>(policy)));
            }
#if !defined(HPX_HAVE_EXECUTOR_COMPATIBILITY)
            else
            {
                HPX_ASSERT(fut.is_ready());
                auto && opt_val = hpx::traits::future_access<future_type>::
                    get_direct_value(std::move(fut));
                auto future = hpx::traits::future_access<future_type>::
                    create_inplace(std::move(*opt_val));
                cont(std::move(future), std::forward<Policy>(policy));
            }
#endif
        }

#if defined(HPX_HAVE_EXECUTOR_COMPATIBILITY)
        template <typename Executor>
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        void attach_exec_v1(Future const& fut, Executor& exec)
        {
            typedef
                typename traits::detail::shared_state_ptr_for<Future>::type
                shared_state_ptr;

            // bind an on_completed handler to this future which will invoke
            // the continuation
            boost::intrusive_ptr<continuation> this_(this);
            void (continuation::*cb)(shared_state_ptr &&, Executor&) =
                &continuation::async_exec_v1<Executor>;

            shared_state_ptr state = traits::detail::get_shared_state(fut);
            typename shared_state_ptr::element_type* ptr = state.get();

            if (ptr == nullptr)
            {
                HPX_THROW_EXCEPTION(no_state,
                    "continuation::attach_exec_v1",
                    "the future to attach has no valid shared state");
            }

            ptr->execute_deferred();
            ptr->set_on_completed(
                util::deferred_call(cb, std::move(this_), std::move(state),
                    std::ref(exec)));
        }
#endif

        template <typename Executor>
        void attach_exec(Future const& future,
            typename std::remove_reference<Executor>::type& exec)
        {
            typedef typename std::decay<Future>::type future_type;
            typedef typename std::remove_reference<Executor>::type
                executor_type;
            typedef typename traits::detail::shared_state_ptr_for<
                future_type>::type shared_state_ptr;

            // bind an on_completed handler to this future which will invoke
            // the continuation
            void (continuation::*cb)(future_type &&, executor_type&) =
                &continuation::async_exec;

            shared_state_ptr state = traits::detail::get_shared_state(future);
            typename shared_state_ptr::element_type* ptr = state.get();

            if (ptr == nullptr && !fut.is_ready())
            {
                HPX_THROW_EXCEPTION(no_state,
                    "continuation::attach_exec",
                    "the future to attach has no valid shared state");
            }

            if (ptr != nullptr)
            {
                ptr->execute_deferred();

                boost::intrusive_ptr<continuation> this_(this);
                auto future = hpx::traits::future_access<future_type>::create(
                    std::move(state));
                ptr->set_on_completed(
                    util::deferred_call(cb, std::move(this_), std::move(future),
                        std::ref(exec)));
            }
#if !defined(HPX_HAVE_EXECUTOR_COMPATIBILITY)
            else
            {
                HPX_ASSERT(fut.is_ready());
                auto && opt_val = hpx::traits::future_access<future_type>::
                    get_direct_value(std::move(fut));
                auto future = hpx::traits::future_access<future_type>::
                    create_inplace(std::move(*opt_val));
                this->*cb(std::move(future), std::ref(exec));
            }
#endif
        }

        template <typename Executor>
        void attach_exec(Future && fut,
            typename std::remove_reference<Executor>::type&& exec)
        {
            typedef typename std::decay<Future>::type future_type;
            typedef typename std::remove_reference<Executor>::type
                executor_type;
            typedef typename traits::detail::shared_state_ptr_for<
                future_type>::type shared_state_ptr;

            // bind an on_completed handler to this future which will invoke
            // the continuation
            void (continuation::*cb)(future_type &&, executor_type&&) =
                &continuation::async_exec;

            shared_state_ptr state = traits::detail::get_shared_state(fut);
            typename shared_state_ptr::element_type* ptr = state.get();

            if (ptr == nullptr && !fut.is_ready())
            {
                HPX_THROW_EXCEPTION(no_state,
                    "continuation::attach_exec",
                    "the future to attach has no valid shared state");
            }

            if (ptr != nullptr)
            {
                ptr->execute_deferred();

                boost::intrusive_ptr<continuation> this_(this);
                auto future = hpx::traits::future_access<future_type>::create(
                    std::move(state));
                ptr->set_on_completed(
                    util::deferred_call(cb, std::move(this_),
                        std::move(future), std::move(exec)));
            }
#if !defined(HPX_HAVE_EXECUTOR_COMPATIBILITY)
            else
            {
                HPX_ASSERT(fut.is_ready());
                auto && opt_val = hpx::traits::future_access<future_type>::
                    get_direct_value(std::move(fut));
                auto future = hpx::traits::future_access<future_type>::
                    create_inplace(std::move(*opt_val));
                this->*cb(std::move(future), std::move(exec));
            }
#endif
        }

    protected:
        bool started_;
        threads::thread_id_type id_;
        typename util::decay<F>::type f_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename ContResult, typename Future, typename Policy, typename F>
    inline typename traits::detail::shared_state_ptr<
        typename continuation_result<ContResult>::type
    >::type
    make_continuation(Future && future, Policy && policy, F && f)
    {
        typedef typename continuation_result<ContResult>::type result_type;
        typedef detail::continuation<Future, F, result_type> shared_state;
        typedef typename shared_state::init_no_addref init_no_addref;

        // create a continuation
        typename traits::detail::shared_state_ptr<result_type>::type p(
            new shared_state(init_no_addref{}, std::forward<F>(f)), false);
        static_cast<shared_state*>(p.get())->attach(
            std::forward<Future>(future), std::forward<Policy>(policy));
        return p;
    }

#if defined(HPX_HAVE_EXECUTOR_COMPATIBILITY)
    template <typename ContResult, typename Future, typename Executor,
        typename F>
    HPX_DEPRECATED(HPX_DEPRECATED_MSG)
    inline typename traits::detail::shared_state_ptr<
        typename continuation_result<ContResult>::type
    >::type
    make_continuation_exec_v1(Future const& future, Executor& exec, F && f)
    {
        typedef typename continuation_result<ContResult>::type result_type;
        typedef detail::continuation<Future, F, result_type> shared_state;
        typedef typename shared_state::init_no_addref init_no_addref;

        // create a continuation
        typename traits::detail::shared_state_ptr<result_type>::type p(
            new shared_state(init_no_addref{}, std::forward<F>(f)), false);
        static_cast<shared_state*>(p.get())->attach_exec_v1(future, exec);
        return p;
    }
#endif

    template <typename ContResult, typename Future, typename Executor,
        typename F>
    inline typename traits::detail::shared_state_ptr<ContResult>::type
    make_continuation_exec(Future && future, Executor && exec, F && f)
    {
        typedef detail::continuation<Future, F, ContResult> shared_state;
        typedef typename shared_state::init_no_addref init_no_addref;

        // create a continuation
        typename traits::detail::shared_state_ptr<ContResult>::type p(
            new shared_state(init_no_addref{}, std::forward<F>(f)), false);
        static_cast<shared_state*>(p.get())->template attach_exec<Executor>(
            std::forward<Future>(future), exec);
        return p;
    }

    template <typename Executor, typename Future, typename F>
    inline typename hpx::traits::future_then_executor_result<
        Executor, typename std::decay<Future>::type, F
    >::type
    then_execute_helper(Executor && exec, F && f, Future && predecessor)
    {
        // simply forward this to executor
        return parallel::execution::then_execute(
            std::forward<Executor>(exec), std::forward<F>(f), predecessor);
    }

    template <typename Executor, typename F, typename Future>
    inline hpx::future<
        typename hpx::util::detail::invoke_deferred_result<
            F, Future
        >::type>
    sync_execute_helper_impl(Executor && exec, F && f, Future && fut,
        std::false_type)
    {
        // simply forward this to executor
        return make_ready_future(parallel::execution::sync_execute(
            std::forward<Executor>(exec), std::forward<F>(f),
            std::forward<Future>(fut)));
    }

    template <typename Executor, typename F, typename Future>
    inline hpx::future<
        typename hpx::util::detail::invoke_deferred_result<
            F, Future
        >::type>
    sync_execute_helper_impl(Executor && exec, F && f, Future && fut,
        std::true_type)
    {
        // simply forward this to executor
        parallel::execution::sync_execute(
            std::forward<Executor>(exec), std::forward<F>(f),
            std::forward<Future>(fut));
        return make_ready_future();
    }

    template <typename Executor, typename F, typename Future>
    inline hpx::future<
        typename hpx::util::detail::invoke_deferred_result<
            F, Future
        >::type>
    sync_execute_helper(Executor && exec, F && f, Future && fut)
    {
        HPX_ASSERT(fut.is_ready());

        using result_type =
            typename hpx::util::detail::invoke_deferred_result<
                    F, Future
                >::type;

        return sync_execute_helper_impl(
            std::forward<Executor>(exec), std::forward<F>(f),
            std::forward<Future>(fut),
            typename std::is_void<result_type>::type{});
    }
}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename ContResult>
    class unwrap_continuation : public future_data<ContResult>
    {
    private:
        template <typename Inner>
        void on_inner_ready(Inner && f)
        {
            try {
                transfer_result<Inner>()(std::move(f), this);
            }
            catch (...) {
                this->set_exception(std::current_exception());
            }
        }

        template <typename Outer>
        void on_outer_ready(Outer && outer)
        {
            typedef typename std::decay<Outer>::type outer_future;
            typedef typename traits::future_traits<outer_future>::type
                inner_future;
            typedef typename traits::detail::shared_state_ptr_for<
                inner_future>::type inner_shared_state_ptr;

            // Bind an on_completed handler to this future which will transfer
            // its result to the new future.
            void (unwrap_continuation::*inner_ready)(inner_future &&) =
                &unwrap_continuation::on_inner_ready<inner_future>;

            try {
                // take by value, as the future will go away immediately
                auto && inner = outer.get();

                inner_shared_state_ptr inner_state =
                    traits::detail::get_shared_state(inner);

                typename inner_shared_state_ptr::element_type* ptr =
                    inner_state.get();
                if (ptr == nullptr && !inner.is_ready())
                {
                    HPX_THROW_EXCEPTION(no_state,
                        "unwrap_continuation<ContResult>::on_outer_ready",
                        "the inner future has no valid shared state");
                }

                if (ptr != nullptr)
                {
                    ptr->execute_deferred();

                    boost::intrusive_ptr<unwrap_continuation> this_(this);
                    ptr->set_on_completed(
                        util::deferred_call(inner_ready, std::move(this_),
                            std::move(inner)));
                }
#if !defined(HPX_HAVE_EXECUTOR_COMPATIBILITY)
                else
                {
                    HPX_ASSERT(inner.is_ready());
                    this->on_inner_ready(std::move(inner));
                }
#endif
            }
            catch (...) {
                this->set_exception(std::current_exception());
            }
        }

    public:
        typedef typename future_data<ContResult>::init_no_addref init_no_addref;

        unwrap_continuation() {}

        unwrap_continuation(init_no_addref no_addref)
          : future_data<ContResult>(no_addref)
        {}

        template <typename Future>
        void attach(Future && fut)
        {
            typedef typename std::decay<Future>::type future_type;
            typedef typename traits::detail::shared_state_ptr_for<
                future_type>::type outer_shared_state_ptr;

            // Bind an on_completed handler to this future which will wait for
            // the inner future and will transfer its result to the new future.
            void (unwrap_continuation::*outer_ready)(Future &&) =
                &unwrap_continuation::on_outer_ready<Future>;

            outer_shared_state_ptr outer_state =
                traits::detail::get_shared_state(fut);
            typename outer_shared_state_ptr::element_type* ptr =
                outer_state.get();

            if (ptr == nullptr && !fut.is_ready())
            {
                HPX_THROW_EXCEPTION(no_state,
                    "unwrap_continuation<ContResult>::attach",
                    "the future has no valid shared state");
            }

            if (ptr != nullptr)
            {
                ptr->execute_deferred();

                boost::intrusive_ptr<unwrap_continuation> this_(this);
                auto future = hpx::traits::future_access<future_type>::create(
                    std::move(outer_state));
                ptr->set_on_completed(
                    util::deferred_call(outer_ready, std::move(this_),
                        std::move(future)));
            }
#if !defined(HPX_HAVE_EXECUTOR_COMPATIBILITY)
            else
            {
                HPX_ASSERT(fut.is_ready());
                auto && opt_val = hpx::traits::future_access<future_type>::
                    get_direct_value(std::forward<Future>(fut));
                auto future = hpx::traits::future_access<future_type>::
                    create_inplace(std::move(*opt_val));
                this->on_outer_ready(std::move(future));
            }
#endif
        }
    };

    template <typename Future>
    inline typename traits::detail::shared_state_ptr<
        typename future_unwrap_result<Future>::result_type>::type
    unwrap_impl(Future && future)
    {
        typedef typename future_unwrap_result<Future>::result_type result_type;

        typedef detail::unwrap_continuation<result_type> shared_state;
        typedef typename shared_state::init_no_addref init_no_addref;

        // create a continuation
        typename traits::detail::shared_state_ptr<result_type>::type p(
            new shared_state(init_no_addref{}), false);
        static_cast<shared_state*>(p.get())->attach(std::forward<Future>(future));
        return p;
    }

    template <typename Result, typename R>
    inline Result unwrap(future<R> && fut)
    {
        typedef typename traits::future_traits<future<R>>::type inner_type;
        typedef
            typename traits::detail::shared_state_ptr_for<inner_type>::type
                inner_shared_ptr_type;

        if (!fut.valid())
        {
            return Result{};
        }

        if (fut.is_ready() && !fut.has_exception())
        {
            inner_type f = fut.get();

            if (f.is_ready() && !f.has_exception())
            {
                return Result{std::move(f)};
            }

            // move the reference count into the returned intrusive_ptr
            return Result{inner_shared_ptr_type(
                traits::future_access<inner_type>::detach_shared_state(
                    std::move(f)),
                false)};
        }

        return Result{unwrap_impl(std::move(fut))};
    }

    template <typename Result, typename Future>
    inline Result unwrap(Future && fut)
    {
        typedef typename traits::future_traits<Future>::type inner_type;
        typedef
            typename traits::detail::shared_state_ptr_for<inner_type>::type
                inner_shared_ptr_type;

        if (!fut.valid())
        {
            return Result{};
        }

        if (fut.is_ready() && !fut.has_exception())
        {
            inner_type f = fut.get();

            if (f.is_ready() && !f.has_exception())
            {
                return Result{std::move(f)};
            }

            // move the reference count into the returned intrusive_ptr
            return Result{inner_shared_ptr_type(
                traits::future_access<inner_type>::detach_shared_state(
                    std::move(f)),
                false)};
        }

        return Result{unwrap_impl(std::forward<Future>(fut))};
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, typename R>
    inline Result share(future<R> && fut)
    {
        if (!fut.valid())
        {
            return Result{};
        }

        bool is_ready = fut.is_ready();
        bool has_exception = fut.has_exception();
        auto shared_state =
            hpx::traits::detail::get_shared_state(std::move(fut));

#if !defined(HPX_HAVE_EXECUTOR_COMPATIBILITY)
        if (is_ready && !shared_state && !has_exception)
        {
            return Result(
                traits::future_access<future<R>>::get_direct_value(fut));
        }
#endif
        return Result{std::move(shared_state)};
    }
}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail
{
    template <typename Future>
    inline typename traits::detail::shared_state_ptr<void>::type
    downcast_to_void(Future& fut, bool addref)
    {
//        if (fut.is_ready() && !fut.has_exception())
//        {
//        }

        typedef typename traits::detail::shared_state_ptr<void>::type
            shared_state_type;
        typedef typename shared_state_type::element_type element_type;

        // same as static_pointer_cast, but with addref option
        return shared_state_type(static_cast<element_type*>(
                traits::detail::get_shared_state(fut).get()
            ), addref);
    }
}}}

#endif
