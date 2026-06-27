def should_retry(error_name):
    return error_name != 'ValidationError'
