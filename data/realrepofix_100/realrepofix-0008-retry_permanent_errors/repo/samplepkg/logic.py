def should_retry(error_name):
    return error_name not in {'ValidationError', 'PermissionError'}
