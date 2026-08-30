interface Props { title: string }
interface Loader { load(): Promise<Props> }
export class Widget extends Component<Props> {
  render(): JSX.Element { return <div>{this.props.title}</div>; }
  mount(): void { return; }
}
export function helper(props: Props): JSX.Element { return <Widget {...props} />; }
export class Panel extends Component<Props> {
  render(): JSX.Element { return <section>{this.props.title}</section>; }
}
export function formatter(props: Props): string { return props.title; }
export const normalize = (value: string): string => value.trim();
