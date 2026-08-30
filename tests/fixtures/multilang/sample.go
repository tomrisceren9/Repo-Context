package sample

type Service struct { Value int }
type Config struct { Retries int }
type Mode int
type Runner interface { Run(value int) string; Stop(force bool) }
func (service *Service) Run(value int) string { return "" }
func (service *Service) Stop(force bool) { }
func helper(value int) int { return value }
func parse(value string) Config { return Config{} }
func validate(value any) bool { return value != nil }
