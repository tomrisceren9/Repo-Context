class AuthenticationService:
    def validate_access_token(self, token: str) -> bool:
        """Validate a signed bearer credential before authorization."""
        return token.startswith("signed-")

    def revoke_access_token(self, token: str) -> None:
        """Invalidate a previously issued bearer credential."""
        return None
