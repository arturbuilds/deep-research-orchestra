from pydantic_settings import SettingsConfigDict, BaseSettings

class Settings(BaseSettings):
    groq_key: str
    tavily_key: str
    ai: str

    model_config = SettingsConfigDict(env_file='.env')

settings = Settings()