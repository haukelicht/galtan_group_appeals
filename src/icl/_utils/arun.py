import asyncio

def run_async(coro):
    """Run async function in both regular Python and Jupyter environments."""
    try:
        # Try to get existing event loop (Jupyter case)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # In Jupyter, use nest_asyncio or create a task
            try:
                import nest_asyncio
                nest_asyncio.apply()
                return asyncio.run(coro)
            except ImportError:
                # Fallback: run in thread pool
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop, regular Python environment
        return asyncio.run(coro)
