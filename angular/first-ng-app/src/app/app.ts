import { Component, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { Header } from './components/header/header';
import { Home } from './home/home';

/* Need to add CheckJS: true to tsconfig.json for
   auto-import */

/* in the dev console use Ctrl-p to search for a function
   then in sources you can place a breakpoint */

/* Use routerLink to link between routes */

/* services are used to encapsulate data,
making http calls, or performing any task that is
not related directly to data rendering
*/

/* if you don't want to share a service
you add to the particular componnt
@Component({
providers: [<nameOfService>]
})

and remove from the service in
@Injectable({
  providedIn: 'root'
})
*/

/*
Note: I had to rename exported class in
services/todos.ts to TodosService and 
accompanying todos.spec.ts
*/

/* Making http calls with Angular
- provide HTTP module/providers in the app
  config using provideHttpClient()
- inject the HttpClient service
*/

/* Angular Directives
allow you to add additional behavior to elements
types:
- components
- attribute directives
- structural directives
*/

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, Header],
  template: `
    <app-header />
    <main>
      <router-outlet />
    </main>
  `,
  styles: [
    `
    main {
      padding: 16px;
    }
    `
  ],
})
export class App {
  protected readonly title = signal('first-ng-app');
}
