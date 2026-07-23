class APIError(Exception):
    """Structured HTTP API error used by the pymss server.

    Args:
        status_code (int): Status code value.
        code (str): Code value.
        message (str): Message value.
        param (str | None, optional): Param value. Defaults to None.
        error_type (str, optional): Error type value. Defaults to 'invalid_request_error'.
    """

    def __init__(self, status_code, code, message, param=None, error_type="invalid_request_error"):
        """Initialize the instance.

        Args:
            status_code (int): Status code value.
            code (str): Code value.
            message (str): Message value.
            param (str | None, optional): Param value. Defaults to None.
            error_type (str, optional): Error type value. Defaults to 'invalid_request_error'.

        Returns:
            None: This method completes for its side effects."""
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.param = param
        self.error_type = error_type
